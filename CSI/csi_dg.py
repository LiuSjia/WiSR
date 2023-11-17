
import numpy as np
from .data_process.signal_process import deal_CSI
import re,os,pickle
import scipy.io as scio
import zarr
from pathlib import Path


widar_gestures=["Push&Pull","Sweep","Clap","Slide","Draw-O(Horizontal)","Draw-Zigzag(Horizontal)"]


def get_widar_csi(root_dir,domain_name):#room_1_user_3_loc_2_ori_3
    index=domain_name.split('_')
    if 'room' in index:
        roomid=index[index.index('room')+1]
        room_ids=[roomid]
    else:
        roomid='1-3'
        room_ids=['1','2','3']

    if 'user' in index:
        userid=index[index.index('user')+1]
        user_ids=[userid]
    else:
        userid='3'
        user_ids=['3']

    if 'ges' in index:
        gesid=index[index.index('ges')+1]
        ges_ids=[gesid]
    else:
        gesid='1-6'
        ges_ids=["Push&Pull","Sweep","Clap","Slide","Draw-O(Horizontal)","Draw-Zigzag(Horizontal)"]

    if 'loc' in index:
        locid=index[index.index('loc')+1]
        loc_ids=[locid]
    else:
        locid='1-5'
        loc_ids=['1','2','3','4','5']

    if 'ori' in index:
        oriid=index[index.index('ori')+1]
        ori_ids=[oriid]
    else:
        oriid='1-5'
        ori_ids=['1','2','3','4','5']

    if 'rx' in index:
        rxid=index[index.index('rx')+1]
        rx_ids=[rxid]
    else:
        rxid='1-6'
        rx_ids=['1','2','3','4','5','6']
    
    data_file_name='room_'+roomid+'_user_'+userid+'_ges_'+gesid+'_loc_'+locid+'_ori_'+oriid+'_rx_'+rxid+'_dealcsidata.pkl'
    data_file=root_dir+'zero_filter_pklfile/'+'room_'+roomid+'/user_'+userid+'/'+data_file_name
    if not os.path.isfile(data_file):
        all_amp=[]
        all_pha=[]
        all_label=[]
        for room in room_ids:
            for user in user_ids:
                for ges in ges_ids:
                    for loc in loc_ids:
                        for ori in ori_ids:
                            for rx in rx_ids:
                                mat_file_name='room_'+room+'_user_'+user+'_ges_'+ges+'_loc_'+loc+'_ori_'+ori+'_rx_'+rx+'_csi.mat'
                                mat_file=root_dir+'matfile/'+mat_file_name
                                #room_1_user_1_ges_Clap_loc_1_ori_1_rx_1_csi.mat
                                if os.path.isfile(mat_file):
                                    mat= scio.loadmat(mat_file)
                                    print('处理：',mat_file)
                                    mat_datas=list(mat.values())[-1][0] 
                                    for csi_data in mat_datas:
                                        amp,pha=deal_CSI(csi_data,IFfilter=True, IFphasani=True,padding_length=2500)
                                        all_amp.append(amp)
                                        all_pha.append(pha)
                                        all_label.append(ges_ids.index(ges))
                                else:
                                    raise ValueError('缺少mat:',mat_file)
        all_amp=np.array(all_amp)
        all_pha=np.array(all_pha)
        all_label=np.array(all_label)

        f=open(data_file,'wb')
        pickle.dump(all_amp,f)
        pickle.dump(all_pha,f)
        pickle.dump(all_label,f)
        f.close()
    else:
        f=open(data_file,'rb')
        print('extracting:',data_file_name)
        all_amp=pickle.load(f)
        all_pha=pickle.load(f)
        all_label=pickle.load(f)
        f.close()

    all_room=np.array([roomid]*all_amp.shape[0],dtype=int)
    all_user=np.array([userid]*all_amp.shape[0],dtype=int)
    all_loc=np.array([locid]*all_amp.shape[0],dtype=int)  
    all_ori=np.array([oriid]*all_amp.shape[0],dtype=int) 
    return all_amp,all_pha,all_label,all_room,all_user,all_loc,all_ori


def get_CSIDA_csi(root_dir,domain_name): #room_1_user_3_loc_2
    index=domain_name.split('_')
    if 'room' in index:
        roomid=index[index.index('room')+1]
        roomid=int(roomid)
    if 'user' in index:
        userid=index[index.index('user')+1]
        userid=int(userid)
    if 'loc' in index:
        locid=index[index.index('loc')+1]
        locid=int(locid)  
        
    data_file=root_dir+'room_'+str(roomid)+'_loc_'+str(locid)+'_user_'+str(userid)+'_CSIDA_amp.pkl'
    if not os.path.isfile(data_file):
        group = zarr.open_group(Path(root_dir).as_posix(), mode="r")
        all_csi = group.csi_data_raw[:]#(2844, 1800, 3, 114)
        all_amp = group.csi_data_amp[:]#(2844, 1800, 3, 114)
        all_pha = group.csi_data_pha[:]
        all_gesture = group.csi_label_act[:]  # 0~5
        room_label = group.csi_label_env[:]  # 0,1
        loc_label = group.csi_label_loc[:]  # 0,1,2
        user_label = group.csi_label_user[:]  # 0,1,2,3,4
        
        index=np.array(range(all_gesture.shape[0]))
        select=index[np.where(room_label==roomid)]
        select=select[np.where(user_label==userid)]
        select=select[np.where(loc_label==locid)]

        all_sel_amp=all_amp[select]
        all_sel_pha=all_pha[select]
        all_sel_csi=all_csi[select]
        all_sel_gesture=all_gesture[select]
        all_sel_room_label=room_label[select]
        all_sel_loc_label=loc_label[select]
        all_sel_user_label=user_label[select]
        
        all_sel_amp=all_sel_amp.transpose(0,2,3,1)#(n, 3, 114, 1800)
        all_sel_pha=all_sel_pha.transpose(0,2,3,1)#(n, 3, 114, 1800)
        all_sel_csi=all_sel_csi.transpose(0,2,3,1)

        f=open(root_dir+'room_'+str(roomid)+'_loc_'+str(locid)+'_user'+str(userid)+'_CSIDA_amp.pkl','wb')
        pickle.dump(all_sel_amp,f)
        f.close()
        f=open(root_dir+'room_'+str(roomid)+'_loc_'+str(locid)+'_user'+str(userid)+'_CSIDA_pha.pkl','wb')
        pickle.dump(all_sel_pha,f)
        f.close()
        f=open(root_dir+'user'+str(userid)+'_CSIDA_label.pkl','wb')
        pickle.dump(all_sel_gesture,f)
        pickle.dump(all_sel_room_label,f)
        pickle.dump(all_sel_loc_label,f)
        pickle.dump(all_sel_user_label,f)
        f.close()
    else:  
        f=open(root_dir+'room_'+str(roomid)+'_loc_'+str(locid)+'_user_'+str(userid)+'_CSIDA_amp.pkl','rb')
        print('extracting:',root_dir+'room_'+str(roomid)+'_loc_'+str(locid)+'_user_'+str(userid)+'_CSIDA_amp.pkl')
        all_sel_amp=pickle.load(f)
        f.close()
        f=open(root_dir+'room_'+str(roomid)+'_loc_'+str(locid)+'_user_'+str(userid)+'_CSIDA_pha.pkl','rb')
        print('extracting:',root_dir+'room_'+str(roomid)+'_loc_'+str(locid)+'_user_'+str(userid)+'_CSIDA_pha.pkl')
        all_sel_pha=pickle.load(f)
        f.close()
        f=open(root_dir+'room_'+str(roomid)+'_loc_'+str(locid)+'_user_'+str(userid)+'_CSIDA_label.pkl','rb')
        all_sel_gesture = pickle.load(f)
        f.close()   

    sel_room=np.array([roomid]*all_sel_amp.shape[0],dtype=int)
    sel_user=np.array([userid]*all_sel_amp.shape[0],dtype=int)
    sel_loc=np.array([locid]*all_sel_amp.shape[0],dtype=int)
    ori_id=np.array([0]*all_sel_amp.shape[0],dtype=int)

    
    return all_sel_amp,all_sel_pha,all_sel_gesture, sel_room,sel_user,sel_loc,ori_id

def get_ARIL_csi(root_dir,domain_name): #train_loc_ test_loc_
    train_amp_data= scio.loadmat(root_dir+"train_data_split_amp.mat")
    train_amp=train_amp_data['train_data']
    train_label=train_amp_data['train_activity_label']
    train_locids=train_amp_data['train_location_label']

    train_pha_data= scio.loadmat(root_dir+"train_data_split_pha.mat")
    train_pha=train_pha_data['train_data']
    assert (train_pha_data['train_activity_label']==train_label).all()
    assert (train_pha_data['train_location_label']==train_locids).all()

    test_amp_data= scio.loadmat(root_dir+"test_data_split_amp.mat")
    test_amp=test_amp_data['test_data']
    test_label=test_amp_data['test_activity_label']
    test_locids=test_amp_data['test_location_label']

    test_pha_data= scio.loadmat(root_dir+"test_data_split_pha.mat")
    test_pha=test_pha_data['test_data']
    assert (test_pha_data['test_activity_label']==test_label).all()
    assert (test_pha_data['test_location_label']==test_locids).all()

    domain_label=domain_name#.split('_')[-1]
    domain_label=int(domain_label)
    
    amp=np.concatenate((train_amp,test_amp),axis=0)
    pha=np.concatenate((train_pha,test_pha),axis=0)
    label=np.concatenate((train_label,test_label),axis=0)
    locids=np.concatenate((train_locids,test_locids),axis=0)
    label=np.squeeze(label) 
    label=label.astype(np.int64)
    locids=np.squeeze(locids)
    index=np.array(range(label.shape[0]))
    select=index[np.where(locids==domain_label)]

    all_amp=amp[select]
    all_pha=pha[select]
    all_label=label[select]
    

    room_id=np.array([0]*all_amp.shape[0],dtype=int)
    user_id=np.array([0]*all_amp.shape[0],dtype=int)
    loc_id=np.array([domain_label]*all_amp.shape[0],dtype=int)
    ori_id=np.array([0]*all_amp.shape[0],dtype=int)
    
    return all_amp,all_pha,all_label,room_id,user_id,loc_id,ori_id
