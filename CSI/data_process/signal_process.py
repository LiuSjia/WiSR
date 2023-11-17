#一些关于信号处理的函数
from scipy import signal,interpolate
import numpy as np
import math
import matplotlib.pyplot as plt
from scipy import signal
#滤波参数

up_cutoff=50
up_order=6
down_cutoff=1
down_order=3
fs=2000



def deal_CSI(csi,IFfilter=True,IFphasani=True,padding_length=None,IFinterp=None,interp_length=None,new_length=None,interp1d_kind='quadratic'):
    #输入形状应该为（3,30,T）
    amp=abs(csi)
    if IFphasani:
        pha=pha_sanitization(csi[0,:,:], csi[1,:,:], csi[2,:,:])
    else:
        pha=np.angle(csi)
    
    if padding_length is not None:
        if padding_length>amp.shape[-1]:
            pad=padding_length-amp.shape[-1]
            zero_array=np.zeros((3,30,pad))
            amp=np.concatenate((amp, zero_array), axis=2)
            pha=np.concatenate((pha, zero_array), axis=2)
        else:
            amp=amp[:,:,0:padding_length]
            pha=pha[:,:,0:padding_length]
    for i in range(amp.shape[0]):
        if IFinterp:
            if interp_length is None:
                interp_length=math.ceil(csi.shape[-1]/1000)*1000
            amp[i,:,:]=data_interp(amp[i,:,:],interp_length,interp1d_kind)
            pha[i,:,:]=data_interp(pha,interp_length,interp1d_kind)
        
        if IFfilter:
            amp[i,:,:]=butter_lowpass(amp[i,:,:], up_cutoff,up_order,down_cutoff, fs, down_order)

        if new_length is not None:
            for j in range(amp.shape[1]):
                amp[i,j,:]=signal.resample(amp[i,j,:],new_length)
                pha[i,j,:]=signal.resample(pha[i,j,:],new_length)
        
    return amp,pha

pi = np.pi

def wrapToPi(data):
    xwrap=np.remainder(data, 2*np.pi)
    mask = np.abs(xwrap)>np.pi
    xwrap[mask] -= 2*np.pi * np.sign(xwrap[mask])
    return xwrap

#相位校准 （两步：解卷绕+线性变换）
# 注意，ant_csi的维度为1*F*T
#https://blog.csdn.net/a_beatiful_knife/article/details/119247331
def pha_sanitization(one_csi, two_csi, three_csi):
    M = 3  # 天线数量3
    N = 30  # 子载波数目30
    T = one_csi.shape[1]  # 总包数
    fi = 312.5 * 2  # 子载波间隔312.5 * 2
    csi_phase = np.zeros((M, N, T))
    csi_phase_diff = np.zeros((M, N, T))
    for t in range(T):  # 遍历时间戳上的CSI包，每根天线上都有30个子载波
        csi_phase[0, :, t] = np.unwrap(np.angle(one_csi[:, t]))
        csi_phase[1, :, t] = np.unwrap(csi_phase[0, :, t] + np.angle(two_csi[:, t] * np.conj(one_csi[:, t])))
        csi_phase[2, :, t] = np.unwrap(csi_phase[1, :, t] + np.angle(three_csi[:, t] * np.conj(two_csi[:, t])))
        ai = np.tile(2 * pi * fi * np.array(range(N)), M)
        bi = np.ones(M * N)
        ci = np.concatenate((csi_phase[0, :, t], csi_phase[1, :, t], csi_phase[2, :, t]))
        A = np.dot(ai, ai)
        B = np.dot(ai, bi)
        C = np.dot(bi, bi)
        D = np.dot(ai, ci)
        E = np.dot(bi, ci)
        rho_opt = (B * E - C * D) / (A * C - B ** 2)
        beta_opt = (B * D - A * E) / (A * C - B ** 2)
        temp = np.tile(np.array(range(N)), M).reshape(M, N)
        csi_phase[:, :, t] = csi_phase[:, :, t] + 2 * pi * fi * temp * rho_opt + beta_opt
        #csi_phase_diff[0, :, t]=csi_phase[0, :, t]-csi_phase[1, :, t]
        #csi_phase_diff[1, :, t]=csi_phase[0, :, t]-csi_phase[2, :, t]
        #csi_phase_diff[2, :, t]=csi_phase[1, :, t]-csi_phase[2, :, t]

    #csi_phase=np.reshape(csi_phase,(M*N, T))
    #csi_phase_diff=np.reshape(csi_phase_diff,(M*N, T))
    
    #for i in range(90):
        #csi_phase_diff[i,:]=wrapToPi(csi_phase_diff[i,:])


    return csi_phase

def data_interp(data,new_length,interp1d_kind):
    x=np.linspace(0,1,data.shape[-1])
    xnew=np.linspace(0,1,new_length)
    data_interp1d=np.empty(data.shape)
    for i in range(data.shape[0]):
        f=interpolate.interp1d(x,data[i,:],kind=interp1d_kind)
        data_interp1d[i,:]=f(xnew)
    print('插值前packets:',data.shape[-1],'插值后packets:', new_length)
    return data_interp1d

#巴特沃斯滤波
def butter_lowpass(data, up_cutoff,up_order,down_cutoff, fs, down_order):
    """
    Design lowpass filter.

    Args:
        - cutoff (float) : the cutoff frequency of the filter.
        - fs     (float) : the sampling rate.
        - order    (int) : order of the filter, by default defined to 5.
    """
    # calculate the Nyquist frequency
    nyq = 0.5 * fs

    # design filter
    low = up_cutoff / nyq
    high = down_cutoff / nyq
    lb, la = signal.butter(up_order, low, btype='low', analog=False)
    hb, ha = signal.butter(down_order, high, btype='high', analog=False)
    filted_data=np.zeros(np.shape(data))
    for i in range(data.shape[0]):
        filted_data[i,:] = signal.filtfilt(lb, la, data[i,:], axis=0)
        filted_data[i,:] = signal.filtfilt(hb, ha, filted_data[i,:], axis=0)
    # returns the filter coefficients: numerator and denominator
    return filted_data

#PCA
def pca_data(data):
    [Ntrx, Nsub,packets] = np.shape(data)
    new_data=np.zeros(np.shape(data))
    for i in range(Ntrx):
        cov_mat=np.cov(data[i,:,:])#天线j的相关矩阵 (30, 30)
        # Calculate eig_val & eig_vec
        eig_val, eig_vec = np.linalg.eig(cov_mat) #(30,) (30, 30)
        # Sort the eig_val & eig_vec
        idx = eig_val.argsort()[::-1] #(30,) 把eig_val中的值按大小排序
        eig_val = eig_val[idx]
        eig_vec = eig_vec[:,idx]
        # Calculate H * eig_vec
        new_data[i,:,:] = data[i,:,:].T.dot(eig_vec).T           
    return new_data

'''
           #PCA分析
           amp_pca=pca_data(amp_filter)
           #只取前n个PCA，3*n，重新组成新的3n*packets矩阵
           amp_seleted=np.concatenate((amp_pca[0,0:n,:],amp_pca[1,0:n,:],amp_pca[2,0:n,:]),axis=0)
           
           amp_var=np.zeros(np.shape(amp_seleted))
           for j in range(Ntx*Nrx*n):
               amp_var[j,:] = tb.VAR(amp_seleted[j,:], timeperiod = window)#还有一个参数nbdev =0或者1，但是对结果好像没影响
               #另一种滑动窗口计算方差的方式，值不同，但是差别不大
               #amp_DF = pd.DataFrame(amp_seleted)
               #b = amp_DF.loc[j].rolling(window).var()#mean()  std()
           #前windows窗口都是nan值，替换成0
           amp_var=np.nan_to_num(amp_var)
           #当前文件最高方差（忽略前几s）
           max_var=np.max(amp_var[:,ignore_points:]) 
           #当前文件最高方差索引
           maxvar_index=np.argmax(amp_var)
           #处理一些最高方差索引超出范围的异常情况
           if maxvar_index>packets:
               a=np.where(amp_var==max_var)
               if len(a)==2:
                   maxvar_index=a[1][0]
               else:
                   for i in range(len(a)):
                       if a[i+1][0]>ignore_points:
                           maxvar_index=a[i+1][0]
                           break
                            
           #动作开始点索引
           start_points=int(maxvar_index-1.25*sample_rate) 
           #动作结束点索引
           end_points=int(maxvar_index+0.75*sample_rate)
           print("start_points:",start_points,"\t\t\t end_points:",end_points)
           print("start_time:",start_points/sample_rate,"\t\t\t end_time:",end_points/sample_rate)
           
           #可视化验证
           plot_subplot(amp_seleted,9,start_points,end_points)
           
           
           #分割amp
           #amp_segment=amp_seleted[:,start_points:end_points]
''' 

#取样
def sampling(amp_matrix,samples):
    amp_sample=np.empty((amp_matrix.shape[0],amp_matrix.shape[1],samples))
    tab=int(amp_matrix.shape[2]/samples)
    for i in range(amp_matrix.shape[0]):
        for j in range(amp_matrix.shape[1]):
            for k in range(samples):
                amp_sample[i,j,k]=amp_matrix[i,j,tab*k]
    return amp_sample