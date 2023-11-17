from scipy import signal,interpolate
import numpy as np
import math
import matplotlib.pyplot as plt
from scipy import signal

up_cutoff=50
up_order=6
down_cutoff=1
down_order=3
fs=2000



def deal_CSI(csi,IFfilter=True,IFphasani=True,padding_length=None,IFinterp=None,interp_length=None,new_length=None,interp1d_kind='quadratic'):
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


def pha_sanitization(one_csi, two_csi, three_csi):
    M = 3  
    N = 30  
    T = one_csi.shape[1]  
    fi = 312.5 * 2  
    csi_phase = np.zeros((M, N, T))
    csi_phase_diff = np.zeros((M, N, T))
    for t in range(T):  
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


    return csi_phase

def data_interp(data,new_length,interp1d_kind):
    x=np.linspace(0,1,data.shape[-1])
    xnew=np.linspace(0,1,new_length)
    data_interp1d=np.empty(data.shape)
    for i in range(data.shape[0]):
        f=interpolate.interp1d(x,data[i,:],kind=interp1d_kind)
        data_interp1d[i,:]=f(xnew)
    print('before packets:',data.shape[-1],'after packets:', new_length)
    return data_interp1d

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
        cov_mat=np.cov(data[i,:,:])
        # Calculate eig_val & eig_vec
        eig_val, eig_vec = np.linalg.eig(cov_mat) #(30,) (30, 30)
        # Sort the eig_val & eig_vec
        idx = eig_val.argsort()[::-1] #(30,) 
        eig_val = eig_val[idx]
        eig_vec = eig_vec[:,idx]
        # Calculate H * eig_vec
        new_data[i,:,:] = data[i,:,:].T.dot(eig_vec).T           
    return new_data



def sampling(amp_matrix,samples):
    amp_sample=np.empty((amp_matrix.shape[0],amp_matrix.shape[1],samples))
    tab=int(amp_matrix.shape[2]/samples)
    for i in range(amp_matrix.shape[0]):
        for j in range(amp_matrix.shape[1]):
            for k in range(samples):
                amp_sample[i,j,k]=amp_matrix[i,j,tab*k]
    return amp_sample