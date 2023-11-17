import itertools
import numpy as np
import pandas as pd

from itertools import combinations

dataset_csi_size={
            'Widar3':{
                'amp':(90, 2500),
                'pha':(90, 2500),
                'amp+pha':(180, 2500),
                'img':(225,225),
                },
            'CSIDA':{
                'amp':(342, 1800),
                'pha':(342, 1800),
                'amp+pha':(684, 1800),
                'img':(225,225),
                },
            'ARIL':{
                'amp':(52, 192),
                'pha':(52, 192),
                'amp+pha':(104, 192),
                'img':(52,52),
                },
            'CSI-Finger':{
                'amp':(90, 4000),
                'pha':(90, 4000),
                'amp+pha':(180, 4000),
                'img':(225,225),
                },
        }


def get_domains(dataset_type,domain_type,ibegin,imax,rxs=None):
    dataset_domain_list={}
    if dataset_type=='CSIDA':
        dataset_domain_list=get_csida_all_domains(ibegin,imax,domain_type,dataset_domain_list)
    if dataset_type=='ARIL':
        dataset_domain_list=get_aril_all_domains(ibegin,imax,domain_type,dataset_domain_list)
    if dataset_type=='Widar3':
        if domain_type in ['loc','ori']:
            dataset_domain_list=get_widar3_all_domains(ibegin,imax,domain_type,dataset_domain_list,rxs=['4'])
        elif domain_type in ['room']:
            dataset_domain_list=get_widar3_all_domains(ibegin,imax,domain_type,dataset_domain_list,rxs=['6'])
        elif domain_type in ['user']:
            dataset_domain_list=get_widar3_all_domains(ibegin,imax,domain_type,dataset_domain_list,rxs=['1'])
        else:
            raise ValueError('wrong')
    return dataset_domain_list



def get_widar3_all_domains(ibegin,imax,domain_type,domain_list,rxs=None):
    i=0
    domain_list['Widar3']=[]
    loc_ids=['1','2','3','4','5']
    ori_ids=['1','2','3','4','5']
    if rxs is None:
        rx_ids=['1','2','3','4','5','6']
    else:
        rx_ids=rxs
    if domain_type=='room':
        user='3'
        rooms=['1','2','3']
        for ori in ori_ids:
            for rx in rx_ids:
                for loc in loc_ids:
                    source_rooms=list(itertools.combinations(rooms, 2))
                    for source_room in source_rooms:
                        i=i+1
                        target_room=set(rooms).difference(set(source_room))
                        source_damains=[]
                        target_damains=[]
                        for room in source_room:
                            source_domain_name='room_'+room+'_user_'+user+'_loc_'+loc+'_ori_'+ori+'_rx_'+rx
                            source_damains.append(source_domain_name)
                        for room in target_room:
                            target_domain_name='room_'+room+'_user_'+user+'_loc_'+loc+'_ori_'+ori+'_rx_'+rx
                            target_damains.append(target_domain_name)
                        dic={
                            'exp_type':'cross_'+domain_type+str(i), 
                            'source_domains':source_damains, 
                            'target_domains':target_damains,
                            'n_subdomains':{'room':len(source_room)},
                            }
                        if i>=ibegin:
                            domain_list['Widar3'].append(dic)
                        
                        if i>=imax:
                            return domain_list
 
    #数据量*
    # widar_room_user_ids={
    #1:['1','2','3','5','10','11','12','13','14','15','16','17'], #5-17* 1/3****  2 *****
    #2:['1','2','3','6'],#1* 3** 2/6*** 
    # 3:['3','7','8','9'] #3-9 *
    # }  
    if domain_type=='ori':
        widar_room_user_ids={
            1:['1','2','3']
            # 3:['3','7','8','9']
            }  
        for room in widar_room_user_ids:
            for loc in loc_ids:
                for rx in rx_ids:
                    for user in widar_room_user_ids[room]:
                        source_oris=list(itertools.combinations(ori_ids, 4))
                        for source_ori in source_oris:
                            i=i+1
                            target_ori=set(ori_ids).difference(set(source_ori))
                            source_damains=[]
                            target_damains=[]
                            for ori in source_ori:
                                source_domain_name='room_'+str(room)+'_user_'+user+'_loc_'+loc+'_ori_'+ori+'_rx_'+rx
                                source_damains.append(source_domain_name)
                            for ori in target_ori:
                                target_domain_name='room_'+str(room)+'_user_'+user+'_loc_'+loc+'_ori_'+ori+'_rx_'+rx
                                target_damains.append(target_domain_name)
                            dic={
                                'exp_type':'cross_'+domain_type+str(i), 
                                'source_domains':source_damains, 
                                'target_domains':target_damains,
                                'n_subdomains':{'ori':len(source_ori)},
                                }
                            if i>=ibegin:
                                domain_list['Widar3'].append(dic)
                            
                            if i>=imax:
                                return domain_list

    if domain_type=='rx':    
        for room in widar_room_user_ids:
            for loc in loc_ids:
                for ori in ori_ids:
                    for user in widar_room_user_ids[room]:
                        source_rxs=list(itertools.combinations(rx_ids, 5))
                        for source_rx in source_rxs:
                            i=i+1
                            target_rx=set(rx_ids).difference(set(source_rx))
                            source_damains=[]
                            target_damains=[]
                            for rx in source_rx:
                                source_domain_name='room_'+str(room)+'_user_'+user+'_loc_'+loc+'_ori_'+ori+'_rx_'+rx
                                source_damains.append(source_domain_name)
                            for rx in target_rx:
                                target_domain_name='room_'+str(room)+'_user_'+user+'_loc_'+loc+'_ori_'+ori+'_rx_'+rx
                                target_damains.append(target_domain_name)
                            dic={
                                'exp_type':'cross_'+domain_type+str(i), 
                                'source_domains':source_damains, 
                                'target_domains':target_damains,
                                'n_subdomains':{'rx':len(source_rx)},
                                }
                            if i>=ibegin:
                                domain_list['Widar3'].append(dic)
                            
                            if i>=imax:
                                return domain_list

    if domain_type=='loc': 
        widar_room_user_ids={
            3:['3','7','8','9'] #3-9 *
            }     
        for ori in ori_ids:
            for rx in rx_ids:
                for room in widar_room_user_ids:
                    for user in widar_room_user_ids[room]:
                        source_locs=list(itertools.combinations(loc_ids, 4))
                        for source_loc in source_locs:
                            i=i+1
                            target_loc=set(loc_ids).difference(set(source_loc))
                            source_damains=[]
                            target_damains=[]
                            for loc in source_loc:
                                source_domain_name='room_'+str(room)+'_user_'+user+'_loc_'+loc+'_ori_'+ori+'_rx_'+rx
                                source_damains.append(source_domain_name)
                            for loc in target_loc:
                                target_domain_name='room_'+str(room)+'_user_'+user+'_loc_'+loc+'_ori_'+ori+'_rx_'+rx
                                target_damains.append(target_domain_name)
                            dic={
                                'exp_type':'cross_'+domain_type+str(i), 
                                'source_domains':source_damains, 
                                'target_domains':target_damains,
                                'n_subdomains':{'loc':len(source_loc)},
                                }
                            if i>=ibegin:
                                domain_list['Widar3'].append(dic)
                            
                            if i>=imax:
                                return domain_list
    if domain_type=='user': 
        widar_room_user_ids={
            3:['3','7','8','9'] #3-9 *
            }  
        for loc in loc_ids:
            for ori in ori_ids:
                for rx in rx_ids:
                    for room in widar_room_user_ids:
                        source_users=list(itertools.combinations(widar_room_user_ids[room], len(widar_room_user_ids[room])-1))
                        for source_user in source_users:
                            i=i+1
                            target_user=set(widar_room_user_ids[room]).difference(set(source_user))
                            source_damains=[]
                            target_damains=[]
                            for user in source_user:
                                source_domain_name='room_'+str(room)+'_user_'+user+'_loc_'+loc+'_ori_'+ori+'_rx_'+rx
                                source_damains.append(source_domain_name)
                            for user in target_user:
                                target_domain_name='room_'+str(room)+'_user_'+user+'_loc_'+loc+'_ori_'+ori+'_rx_'+rx
                                target_damains.append(target_domain_name)
                            dic={
                                'exp_type':'cross_'+domain_type+str(i), 
                                'source_domains':source_damains, 
                                'target_domains':target_damains,
                                'n_subdomains':{'user':len(source_user)},
                                }
                            if i>=ibegin:
                                domain_list['Widar3'].append(dic)
                            
                            if i>=imax:
                                return domain_list
    return domain_list


def get_csida_all_domains(ibegin,imax,domain_type,domain_list):##room 0-1 user 0-4 loc 0-2
    i=0
    domain_list['CSIDA']=[]
    loc_ids=['0','1','2']
    user_ids=['0','1','2','3','4']
    rooms=['0','1']
    common_locs=['0','1']
    
    csida_room_loc_ids={
    '0':['0','1','2'],#,'5','11','12','13','14','15','16','17'],
    '1':['0','1'],
    }  
    if domain_type=='room':
        for user in user_ids:
            for loc in common_locs:
                source_rooms=list(itertools.combinations(rooms, 1))
                for source_room in source_rooms:
                    i=i+1
                    target_room=set(rooms).difference(set(source_room))
                    source_damains=[]
                    target_damains=[]
                    for room in source_room:
                        source_domain_name='room_'+str(room)+'_user_'+user+'_loc_'+loc
                        source_damains.append(source_domain_name)
                    for room in target_room:
                        target_domain_name='room_'+str(room)+'_user_'+user+'_loc_'+loc
                        target_damains.append(target_domain_name)
                    dic={
                        'exp_type':'cross_'+domain_type+str(i), 
                        'source_domains':source_damains, 
                        'target_domains':target_damains,
                        'n_subdomains':{'room':len(source_room)},
                        }
                    if i>=ibegin:
                        domain_list['CSIDA'].append(dic)
                    
                    if i>=imax:
                        return domain_list
    if domain_type=='loc':    
        for room in rooms:
            for user in user_ids:
                if room=='0':
                    loc_ids=csida_room_loc_ids[room]
                    source_locs=list(itertools.combinations(loc_ids, 2))
                else:
                    loc_ids=csida_room_loc_ids[room]
                    source_locs=list(itertools.combinations(loc_ids, 1))
                for source_loc in source_locs:
                    i=i+1
                    target_loc=set(loc_ids).difference(set(source_loc))
                    source_damains=[]
                    target_damains=[]
                    for loc in source_loc:
                        source_domain_name='room_'+str(room)+'_user_'+user+'_loc_'+loc
                        source_damains.append(source_domain_name)
                    for loc in target_loc:
                        target_domain_name='room_'+str(room)+'_user_'+user+'_loc_'+loc
                        target_damains.append(target_domain_name)
                    dic={
                        'exp_type':'cross_'+domain_type+str(i), 
                        'source_domains':source_damains, 
                        'target_domains':target_damains,
                        'n_subdomains':{'loc':len(source_loc)},
                        }
                    if i>=ibegin:
                        domain_list['CSIDA'].append(dic)
                    
                    if i>=imax:
                        return domain_list
    if domain_type=='user': 
        for room in rooms:
            if room=='0':
                loc_ids=csida_room_loc_ids[room]
            else:
                loc_ids=csida_room_loc_ids[room]
            for loc in loc_ids:
                source_users=list(itertools.combinations(user_ids, 4))
                for source_user in source_users:
                    i=i+1
                    target_user=set(user_ids).difference(set(source_user))
                    source_damains=[]
                    target_damains=[]
                    for user in source_user:
                        source_domain_name='room_'+str(room)+'_user_'+user+'_loc_'+loc
                        source_damains.append(source_domain_name)
                    for user in target_user:
                        target_domain_name='room_'+str(room)+'_user_'+user+'_loc_'+loc
                        target_damains.append(target_domain_name)
                    dic={
                        'exp_type':'cross_'+domain_type+str(i), 
                        'source_domains':source_damains, 
                        'target_domains':target_damains,
                        'n_subdomains':{'user':len(source_user)},
                        }
                    if i>=ibegin:
                        domain_list['CSIDA'].append(dic)
                    
                    if i>=imax:
                        return domain_list

    if domain_type=='room_user': 
        for room in rooms:
            if room=='0':
                loc_ids=csida_room_loc_ids[room]
            else:
                loc_ids=csida_room_loc_ids[room]
            for user in user_ids: 
                for loc in common_locs:
                    i=i+1
                    target_domain_name='room_'+str(room)+'_user_'+user+'_loc_'+loc
                    source_rooms=set(rooms).difference(set(room))
                    source_users=set(user_ids).difference(set(user))
                    source_damains=[]
                    for source_user in source_users:
                        for source_room in source_rooms:
                            source_domain_name='room_'+str(source_room)+'_user_'+str(source_user)+'_loc_'+loc
                            source_damains.append(source_domain_name)
                    dic={
                        'exp_type':'cross_'+domain_type+str(i), 
                        'source_domains':source_damains, 
                        'target_domains':[target_domain_name],
                        'n_subdomains':{'room':len(source_rooms),
                                         'user':len(source_users)
                                        },
                        }
                    if i>=ibegin:
                        domain_list['CSIDA'].append(dic) 
                    if i>=imax:
                        return domain_list
    if domain_type=='room_loc': 
        for room in rooms:
            if room=='0':
                loc_ids=csida_room_loc_ids[room]          
            else:
                loc_ids=csida_room_loc_ids[room]
            for user in user_ids: 
                for loc in loc_ids:
                    i=i+1
                    target_domain_name='room_'+str(room)+'_user_'+user+'_loc_'+loc
                    source_rooms=set(rooms).difference(set(room))
                    for source_room in source_rooms:
                        source_locs=set(csida_room_loc_ids[source_room]).difference(set(loc))
                        n_loc=len(source_locs)
                        source_damains=[]
                        for source_loc in source_locs:
                            source_domain_name='room_'+str(source_room)+'_user_'+user+'_loc_'+source_loc
                            source_damains.append(source_domain_name)
                    dic={
                        'exp_type':'cross_'+domain_type+str(i), 
                        'source_domains':source_damains, 
                        'target_domains':[target_domain_name],
                        'n_subdomains':{'room':len(source_rooms),
                                         'loc':len(source_locs),
                                        }
                        }
                    if i>=ibegin:
                        domain_list['CSIDA'].append(dic)
                    
                    if i>=imax:
                        return domain_list

    if domain_type=='user_loc':    
        for room in rooms:
            if room=='0':
                loc_ids=csida_room_loc_ids[room]
                n_loc=2
            else:
                loc_ids=csida_room_loc_ids[room]
                n_loc=1
            for user in user_ids: 
                for loc in loc_ids:
                    i=i+1
                    target_domain_name='room_'+str(room)+'_user_'+user+'_loc_'+loc
                    source_users=set(user_ids).difference(set(user))
                    source_locs=set(loc_ids).difference(set(loc))
                    source_damains=[]
                    for source_loc in source_locs:
                        for source_user in source_users:
                            source_domain_name='room_'+room+'_user_'+source_user+'_loc_'+source_loc
                            source_damains.append(source_domain_name)
                    dic={
                        'exp_type':'cross_'+domain_type+str(i), 
                        'source_domains':source_damains, 
                        'target_domains':[target_domain_name],
                        'n_subdomains':{'loc':len(source_locs),
                                        'user':len(source_users)},
                        }
                    if i>=ibegin:
                        domain_list['CSIDA'].append(dic)
                    
                    if i>=imax:
                        return domain_list

    if domain_type=='room_user_loc':    
        for room in rooms:
            if room=='0':
                loc_ids=csida_room_loc_ids[room]
                n_loc=2
            else:
                loc_ids=csida_room_loc_ids[room]
                n_loc=1
            for user in user_ids: 
                for loc in loc_ids:
                    i=i+1
                    target_domain_name='room_'+str(room)+'_user_'+user+'_loc_'+loc
                    source_users=set(user_ids).difference(set(user))
                    source_rooms=set(rooms).difference(set(room))
                    source_damains=[]
                    for source_room in source_rooms:
                        source_locs=set(csida_room_loc_ids[source_room]).difference(set(loc))
                        n_loc=len(source_locs)
                        for source_loc in source_locs:
                            for source_user in source_users:
                                source_domain_name='room_'+source_room+'_user_'+source_user+'_loc_'+source_loc
                                source_damains.append(source_domain_name)
                    dic={
                        'exp_type':'cross_'+domain_type+str(i), 
                        'source_domains':source_damains, 
                        'target_domains':[target_domain_name],
                        'n_subdomains':{'room':len(source_rooms),
                                         'loc':len(source_locs),
                                        'user':len(source_users)
                                        },
                        }
                    if i>=ibegin:
                        domain_list['CSIDA'].append(dic)
                    
                    if i>=imax:
                        return domain_list
    
    return domain_list

def get_aril_all_domains(ibegin,imax,domain_type,domain_list):##room 0-1 user 0-4 loc 0-2
    i=0
    domain_list['ARIL']=[]
    loc_ids=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]
    for a in range(len(loc_ids)):
        i=i+1
        target_damains=[str(loc_ids[a])]  
        source_damains=[]
        for b in range(len(loc_ids)):
            if b!=a:
                source_damains.append(str(loc_ids[b]))       
        dic={
            'exp_type':'cross_'+domain_type+str(i), 
            'source_domains':source_damains, 
            'target_domains':target_damains,
            'n_subdomains':{'loc':15},
            }
        if i>=ibegin:
            domain_list['ARIL'].append(dic)
        
        if i>=imax:
            return domain_list
    return domain_list

