import torch
import numpy as np

from datasets import datasets
from lib import misc


def get_dataset(args, hparams):
    
    dataset=vars(datasets)[args.dataset](args)
    train_dataset = vars(datasets)[args.dataset](args, 'source').datasets
    test_dataset = vars(datasets)[args.dataset](args, 'target').datasets
    train_splits = []
    test_splits = []
    for env in train_dataset:
        env=shuffle_dataset(env,args.seed)
        in_type="train"
        #set_transfroms(env, in_type, hparams, algorithm_class)
        weights = None
        train_splits.append((env, weights))
    for env in test_dataset:
        env=shuffle_dataset(env,args.seed)
        out_type = "test"
        #set_transfroms(env, out_type, hparams, algorithm_class)
        weights = None
        test_splits.append((env, weights))


    return dataset, train_splits, test_splits


class _SplitDataset(torch.utils.data.Dataset):
    """Used by split_dataset"""

    def __init__(self, underlying_dataset, keys):
        super(_SplitDataset, self).__init__()
        self.underlying_dataset = underlying_dataset
        self.keys = keys
        self.transforms = {}

        self.direct_return = isinstance(underlying_dataset, _SplitDataset)

    def __getitem__(self, key):
        if self.direct_return:
            return self.underlying_dataset[self.keys[key]]

        x, y,d = self.underlying_dataset[self.keys[key]]
        #ret = {x,y}#{"y": y}

        #for key, transform in self.transforms.items():
            #ret[key] = transform(x)

        return (x, y,d),key #ret

    def __len__(self):
        return len(self.keys)


def split_dataset(dataset, n, seed=0):
    """
    Return a pair of datasets corresponding to a random split of the given
    dataset, with n datapoints in the first dataset and the rest in the last,
    using the given random seed
    """
    assert n <= len(dataset)
    keys = list(range(len(dataset)))
    np.random.RandomState(seed).shuffle(keys)
    keys_1 = keys[:n]
    keys_2 = keys[n:]
    return _SplitDataset(dataset, keys_1), _SplitDataset(dataset, keys_2)


def shuffle_dataset(dataset,seed=0):
    """
    Return a pair of datasets corresponding to a random split of the given
    dataset, with n datapoints in the first dataset and the rest in the last,
    using the given random seed
    """
    keys = list(range(len(dataset)))
    np.random.RandomState(seed).shuffle(keys)
    return _SplitDataset(dataset, keys)
