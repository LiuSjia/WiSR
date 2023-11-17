# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import torch
from torch.utils.data import TensorDataset

import numpy as np
from CSI.csi_domainset import dataset_csi_size
from CSI.csi_dg import get_widar_csi,get_CSIDA_csi,get_ARIL_csi


DATASETS = [
    "CSIDA"
]

class MultipleDomainDataset:
    N_STEPS = 5001           # Default, subclasses may override
    CHECKPOINT_FREQ = 100    # Default, subclasses may override
    N_WORKERS = 8            # Default, subclasses may override
    ENVIRONMENTS = None      # Subclasses should override
    INPUT_SHAPE = None       # Subclasses should override

    def __getitem__(self, index):
        return self.datasets[index]

    def __len__(self):
        return len(self.datasets)

class CSIIterDataset(torch.utils.data.IterableDataset):
    def __init__(self,args,domain,data_type) -> None:
        self.data_type=data_type
        self.data, self.labels=self.get_csidataset(args,domain)
        

    def get_csidataset(self, args,domain):
        dataset_dir=args.data_dir+args.csidataset
        if args.csidataset=='Widar3':
            dataset_dir=dataset_dir+"/CSI/"
            amp,pha,labels,roomid,userid,locid,oriid=get_widar_csi(dataset_dir,domain) #(n, 3, 30, 2500)
        elif args.csidataset=='CSIDA': 
            dataset_dir=dataset_dir+"/CSI_301/"
            amp,pha,labels,roomid,userid,locid,oriid=get_CSIDA_csi(dataset_dir,domain)#(n, 3, 114, 1800)
        elif args.csidataset=='ARIL': #'Widar3/CSI/'   
            dataset_dir=dataset_dir+"/"
            amp,pha,labels,roomid,userid,locid,oriid=get_ARIL_csi(dataset_dir,domain)#(n, 52,192)
            amp=amp.reshape(amp.shape[0],1,amp.shape[1],amp.shape[2])
            pha=pha.reshape(pha.shape[0],1,pha.shape[1],pha.shape[2])

        input_datatype=args.data_type
        if input_datatype=='amp':
            data=amp#(n, 3, 30, 2500)
        elif input_datatype=='pha':
            data=pha#(n, 3, 30, 2500)
        elif input_datatype=='amp+pha':
            data=np.concatenate((amp,pha),axis=2)#(n, 3, 60, 2500)
        else:
            raise ValueError('wrong type')
        data=data.reshape(data.shape[0],data.shape[1]*data.shape[2],data.shape[3]) #(n, 180, 2500)

        return data,labels

    def __getitem__(self, index: int):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        data, label = self.data[index], int(self.labels[index])

        return data, label

    def __len__(self) -> int:
        return len(self.data)

class MultipleEnvironmentCSI(MultipleDomainDataset):
    def __init__(self, args, data_type,
                 num_classes,dataset_transform):
        super().__init__()
        self.datasets = []

        if data_type is not None:
            if data_type=='source':
                domains=args.source_domains
            if data_type=='target':
                domains=args.target_domains
            for i in range(len(domains)):
                dataset=CSIIterDataset(args,domains[i],data_type)
                x=dataset.data
                y=dataset.labels
                d=np.array([i]*x.shape[0],dtype=int)

                x=torch.tensor(x, dtype=torch.float)#torch.from_numpy(data)#浅拷贝，共享内存空间
                y=torch.tensor(y)
                d=torch.tensor(d)
                self.datasets.append(dataset_transform(x,y,d))

        self.input_shape = dataset_csi_size[args.csidataset][args.data_type]
        self.num_classes = num_classes

class CSI(MultipleEnvironmentCSI):

    def __init__(self, args, data_type=None):
        super(CSI, self).__init__(args, data_type, 
                 6, self.dataset_transform)
         
    def dataset_transform(self, x, y,domain):
        return TensorDataset(x, y,domain)

