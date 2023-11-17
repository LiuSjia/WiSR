#从dat文件中提取CSI
from __future__ import print_function, absolute_import, division

import codecs
import pandas as pd
import numpy as np
import math
import os
import matplotlib.pyplot as plt
import scipy.io as scio


def read_bfee(in_bytes, i, df_data):
    df_data.loc[i, 'cell'] = i + 1  # 第几个包
    # NIC网卡1MHz时钟
    df_data.loc[i, 'timestamp_low'] = in_bytes[0] + (in_bytes[1] << 8) + (in_bytes[2] << 16) + (in_bytes[3] << 24)
    # 驱动记录并发送到用户控件的波束测量值的总数
    df_data.loc[i, 'bfee_count'] = in_bytes[4] + (in_bytes[5] << 8)

    df_data.loc[i, 'Nrx'] = in_bytes[8]  # 接收端天线数量
    df_data.loc[i, 'Ntx'] = in_bytes[9]  # 发送端天线数量
    # 接收端NIC测量出的RSSI值
    df_data.loc[i, 'rssi_a'] = in_bytes[10]
    df_data.loc[i, 'rssi_b'] = in_bytes[11]
    df_data.loc[i, 'rssi_c'] = in_bytes[12]

    df_data.loc[i, 'noise'] = in_bytes[13].astype(np.int8)
    df_data.loc[i, 'agc'] = in_bytes[14]
    df_data.loc[i, 'rate'] = in_bytes[18] + (in_bytes[19] << 8)  # 发包频率

    # perm 展示NIC如何将3个接收天线的信号排列在3个RF链上
    # [1,2,3] 表示天线A被发送到RF链A，天线B--> RF链B，天线C--> RF链C
    # [1,3,2] 表示天线A被发送到RF链A，天线B--> RF链C，天线C--> RF链B
    perm = [1, 1, 1]
    antenna_sel = in_bytes[15]
    perm[0] = (antenna_sel & 0x3) + 1
    perm[1] = ((antenna_sel >> 2) & 0x3) + 1
    perm[2] = ((antenna_sel >> 4) & 0x3) + 1
    df_data.loc[i, 'perm'] = perm

    # csi
    leng = in_bytes[16] + (in_bytes[17] << 8)
    calc_len = int((30 * (in_bytes[8] * in_bytes[9] * 8 * 2 + 3) + 7) / 8)
    index = 0
    payload = in_bytes[20:]
    n_tx = in_bytes[9]
    csi = np.empty([n_tx, 3, 30], dtype=complex)
    if leng == calc_len:
        for k in range(30):
            index += 3
            remainder = index % 8
            a = []
            if index>=payload.shape[0]:
                index=payload.shape[0]-1
            for j in range(in_bytes[8] * in_bytes[9]):
                tmp_r = ((payload[int(index / 8)] >> remainder) | (
                        payload[int(index / 8 + 1)] << (8 - remainder))).astype(np.int8)
                tmp_i = ((payload[int(index / 8 + 1)] >> remainder) | (
                        payload[int(index / 8 + 2)] << (8 - remainder))).astype(np.int8)
                a.append(complex(tmp_r, tmp_i))
                index += 16
                j += 1

            csi[:, perm[0] - 1, k] = a[:n_tx]
            csi[:, perm[1] - 1, k] = a[n_tx:2 * n_tx]
            csi[:, perm[2] - 1, k] = a[2 * n_tx:3 * n_tx]
            k += 1
    df_data.loc[i, 'csi'] = csi
    return df_data


def read_bf_file(filename, offset=0):
    cur = offset
    count = 0
    data = pd.DataFrame(
        columns=['cell', 'timestamp_low', 'bfee_count', 'Nrx', 'Ntx', 'rssi_a', 'rssi_b', 'rssi_c', 'noise', 'agc',
                 'perm', 'rate', 'csi'])
    triangle = [1, 3, 6]
    f = open(filename, 'rb')
    f.seek(cur)
    data_len = os.path.getsize(filename)
    bytes_ = None
    while cur < (data_len - 3):
        field_len = int(codecs.encode(f.read(2), 'hex'), 16)
        code = int(codecs.encode(f.read(1), 'hex'), 16)
        cur = cur + 3

        if code == 187:
            bytes_ = np.fromfile(f, np.uint8, count=field_len - 1)
            cur = cur + field_len - 1
            if len(bytes_) != field_len - 1:
                f.close()
        else:
            f.seek(field_len - 1, 1)
            cur = cur + field_len - 1

        if code == 187:
            data = read_bfee(in_bytes=bytes_, i=count, df_data=data)
            perm = data.loc[count, 'perm']
            n_rx = data.loc[count, 'Nrx']

            if sum(perm) == triangle[n_rx - 1]:
                count = count + 1
    f.close()
    return data, cur


def dbinv(x):
    ret = 10 ** (x / 10)
    return ret


def get_total_rss(dic):
    rssi_mag = 0
    if dic['rssi_a'] != 0:
        rssi_mag = rssi_mag + dbinv(dic['rssi_a'])
    if dic['rssi_b'] != 0:
        rssi_mag = rssi_mag + dbinv(dic['rssi_b'])
    if dic['rssi_c'] != 0:
        rssi_mag = rssi_mag + dbinv(dic['rssi_c'])

    ret = 10 * math.log10(rssi_mag) - 44 - dic['agc']

    return ret


def get_scale_csi(dic):
    csi = dic['csi']
    csi_conj = csi.conjugate()
    csi_sq = np.multiply(csi, csi_conj).real
    csi_pwr = sum(sum(sum(csi_sq[:])))
    rssi_pwr = dbinv(get_total_rss(dic))
    scale = rssi_pwr / (csi_pwr / 30)

    if dic['noise'] == -127:
        noise_db = -92
    else:
        noise_db = dic['noise']

    thermal_noise_pwr = dbinv(noise_db)
    quant_error_pwr = scale * (dic['Nrx'] * dic['Ntx'])
    total_noise_pwr = thermal_noise_pwr + quant_error_pwr

    ret = csi * math.sqrt(scale / total_noise_pwr)

    if dic['Ntx'] == 2:
        ret = ret * math.sqrt(2)
    elif dic['Ntx'] == 3:
        ret = ret * math.sqrt(dbinv(4.5))

    return ret

'''
感觉作用不大，真实相位和真实相位差趋势一样，真实相位差反而更粗糙，有些动作的趋势变化不明显
#true_pha=np.empty([3,30,packets]) #真实相位
#true_pha_diff=np.empty([3,30,packets]) #真实相位差

#csi_phase1 = get_true_phase(csi[0,0,:], -1)#-1的话是具体子载波的值(30,)
#csi_phase2 = get_true_phase(csi[0,1,:], -1)#-1的话是具体子载波的值(30,)
#csi_phase3 = get_true_phase(csi[0,2,:], -1)#-1的话是具体子载波的值(30,)
#csi_phase = get_true_phase(csi, -2)#-2是所有子载波的值(1, 90)


#true_pha[0,:,i] = get_true_phase(csi[0,0,:], -1)
#true_pha[1,:,i] = get_true_phase(csi[0,1,:], -1)
#true_pha[2,:,i] = get_true_phase(csi[0,2,:], -1)
        
      #真实相位差
    true_pha_diff[0]=true_pha[0]-true_pha[1]
    true_pha_diff[1]=true_pha[0]-true_pha[2]
    true_pha_diff[2]=true_pha[1]-true_pha[2]
    true_pha_diff=np.unwrap(true_pha_diff)  

def get_true_phase(csi, index):
        """
        :param csi: csi data
        :param index: 
         if index is -1 return all true phase,如果index为-1返回所有真相位
         else if index is 0 return a pair antenna,否则如果指数为0返回一对天线，
         else return the tx-rx-subcarrier number true phase.否则返回tx-rx子载波数真相位。
        :return: true phase data
        """
        import math
        csi_phase = np.angle(csi)#(1, 3, 30)
        temp = np.zeros(30)
        recycle = 0
        k_index_i = np.array([-28, -26, -24, -22, -20, -18, -16, -14, -12, -10, -8, -6, -4, -2, -1,
                              1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 28]).T
        # k_index_i = np.array([-58, -54, -50, -46, -42, -38, -34, -30, -26, -22, -18, -14, -10, -6, -2,
        #                       2, 6, 10, 14, 18, 22, 26, 30, 34, 38, 42, 46, 50, 54, 58]).T
        if index >= -1:
            temp[0] = csi_phase[0]
            for t_i in range(1, 30):
                if csi_phase[t_i] - csi_phase[t_i - 1] > math.pi:
                    recycle = recycle + 1
                temp[t_i] = csi_phase[t_i] - recycle * 2 * math.pi
            csi_phase = temp.T
            a = (csi_phase[29] - csi_phase[0]) / 56
            b = np.mean(csi_phase)
            true_phase = csi_phase - a * k_index_i - b
            if index == -1:
                return true_phase  # 30 subcarriers
            else:
                return true_phase[index]  # 1 subcarrier
        else:
            [tx, rx, s] = np.shape(csi_phase)
            true_phase = np.zeros((1, tx * rx * s))
            for i in range(tx):
                for j in range(rx):
                    temp[0] = csi_phase[i][j][0]
                    for t_i in range(1, 30):
                        if csi_phase[i][j][t_i] - csi_phase[i][j][t_i - 1] > math.pi:
                            recycle = recycle + 1
                        temp[t_i] = csi_phase[i][j][t_i] - recycle * 2 * math.pi
                    csi_phase[i][j] = temp.T
                    a = (csi_phase[i][j][29] - csi_phase[i][j][0]) / 56
                    b = np.mean(csi_phase[i][j])
                    true_phase[0, (90 * i + 30 * j):(90 * i + 30 * (j + 1))] = csi_phase[i][j] - a * k_index_i - b
            return true_phase
'''


def extract_CSI_csv(filename):
#if __name__ == '__main__':
    #filename="C:/Users/LiuSJ/Desktop/Experiment/WiSTS/data/csv/100-sit1.dat.csv"
    data = pd.read_csv(filename, header=None).values
    packets=data.shape[0] 
    print('CSI packets:', packets) 
    amp=np.empty([3,30,packets]) #振幅
    pha=np.empty([3,30,packets]) #相位
    pha_diff=np.empty([3,30,packets]) #相位差
    amp[0,:,:] = data[:,1:31].T
    amp[1,:,:] = data[:,31:61].T
    amp[2,:,:] = data[:,61:91].T
    pha[0,:,:] = data[:,91:121].T
    pha[1,:,:] = data[:,121:151].T
    pha[2,:,:] = data[:,151:181].T
    pha_diff[0]=pha[0]-pha[1]
    pha_diff[1]=pha[0]-pha[2]
    pha_diff[2]=pha[1]-pha[2]
    pha_diff=np.unwrap(pha_diff)
    return amp,pha,pha_diff

def extract_CSI_mat(filename):
#if __name__ == '__main__':
    #filename="C:/Users/LiuSJ/Desktop/Experiment/WiSTS/data/csv/100-sit1.dat.csv"
    csi_data= scio.loadmat('D:/Dataset/CSI-Data/dat/'+room+r"_"+user+r"_"+activity+r'_CSI.mat')
    csi=list(csi_data.values())[-1][:,1]#0是文件名，1是CSI值 (50,)
    return csi


def extract_CSI_dat(filename):
#if __name__ == '__main__':
    #filename = r'C:\Users\LiuSJ\Desktop\Google云端硬盘\WiSTS\Experiment\matlab\data\lsj\1-5s-sist-1.dat'
    offset = 0
    csi_trace, offset = read_bf_file(filename, offset) #dataframe
    #print(csi_trace.shape) #(packets,13) dataframe
    packets=csi_trace.shape[0]
    print('CSI packets:', packets) 
    csi_e = get_scale_csi(csi_trace.loc[1])
    [Ntx, Nrx, Nsub] = np.shape(csi_e) #发射天线，接收天线，子载波数量 
    csi_data=np.empty([Ntx,Nrx,Nsub,packets],dtype = complex)
    #pha_diff=np.empty([3,30,packets]) #相位差
    timestamp=np.empty(packets) #时间戳
    
    for i in range(packets):
        for j in range(Ntx):
            for k in range(Nrx):
                csi_entry = csi_trace.loc[i] #第i个CSI包,(13,)
                csi = get_scale_csi(csi_entry) #Ntx, Nrx, Nsub complex double，numpy.ndarray
                timestamp[i]=csi_trace.loc[i, 'timestamp_low']
                csi_data[j,k,:,i] = csi[j,k,:]
            #timestamp_seq[i]=csi_entry.timestamp_low
        
    #幅度值  
    #amp = np.abs(csi)
    '''
    #相位值
    pha = np.angle(RX_antenna)
    #相位差
    pha_diff[0]=pha[0]-pha[1]
    pha_diff[1]=pha[0]-pha[2]
    pha_diff[2]=pha[1]-pha[2]
    pha_diff=np.unwrap(pha_diff)
    
    
    plt.plot(np.unwrap(pha[0,0,:]))
    plt.title('Phase')
    plt.show()
   
    plt.plot(pha_diff[0,0,:])
    plt.title('Phase Difference')
    plt.show()
    
    plt.plot(amp[0,0,:])
    plt.title('Amplitude')
    plt.show()
    '''
    return csi_data