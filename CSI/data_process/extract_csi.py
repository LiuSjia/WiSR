from __future__ import print_function, absolute_import, division

import codecs
import pandas as pd
import numpy as np
import math
import os
import matplotlib.pyplot as plt
import scipy.io as scio


def read_bfee(in_bytes, i, df_data):
    df_data.loc[i, 'cell'] = i + 1  
    df_data.loc[i, 'timestamp_low'] = in_bytes[0] + (in_bytes[1] << 8) + (in_bytes[2] << 16) + (in_bytes[3] << 24)
    df_data.loc[i, 'bfee_count'] = in_bytes[4] + (in_bytes[5] << 8)

    df_data.loc[i, 'Nrx'] = in_bytes[8]  
    df_data.loc[i, 'Ntx'] = in_bytes[9]  

    df_data.loc[i, 'rssi_a'] = in_bytes[10]
    df_data.loc[i, 'rssi_b'] = in_bytes[11]
    df_data.loc[i, 'rssi_c'] = in_bytes[12]

    df_data.loc[i, 'noise'] = in_bytes[13].astype(np.int8)
    df_data.loc[i, 'agc'] = in_bytes[14]
    df_data.loc[i, 'rate'] = in_bytes[18] + (in_bytes[19] << 8) 

    
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


def extract_CSI_csv(filename):
    data = pd.read_csv(filename, header=None).values
    packets=data.shape[0] 
    print('CSI packets:', packets) 
    amp=np.empty([3,30,packets]) 
    pha=np.empty([3,30,packets]) 
    pha_diff=np.empty([3,30,packets]) 
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
    csi_data= scio.loadmat('D:/Dataset/CSI-Data/dat/'+room+r"_"+user+r"_"+activity+r'_CSI.mat')
    csi=list(csi_data.values())[-1][:,1]
    return csi


def extract_CSI_dat(filename):
    offset = 0
    csi_trace, offset = read_bf_file(filename, offset) #dataframe
    #print(csi_trace.shape) #(packets,13) dataframe
    packets=csi_trace.shape[0]
    print('CSI packets:', packets) 
    csi_e = get_scale_csi(csi_trace.loc[1])
    [Ntx, Nrx, Nsub] = np.shape(csi_e) 
    csi_data=np.empty([Ntx,Nrx,Nsub,packets],dtype = complex)
    timestamp=np.empty(packets) 
    
    for i in range(packets):
        for j in range(Ntx):
            for k in range(Nrx):
                csi_entry = csi_trace.loc[i] 
                csi = get_scale_csi(csi_entry) 
                timestamp[i]=csi_trace.loc[i, 'timestamp_low']
                csi_data[j,k,:,i] = csi[j,k,:]
        

    return csi_data