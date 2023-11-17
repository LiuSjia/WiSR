# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import numpy as np


def _hparams(algorithm, dataset, random_state):
    hparams = {}

    hparams["optimizer"] = ("sgd", "sgd")
    hparams["lr"] = (0.05, 10 ** random_state.uniform(-5, -3.5))
    hparams["weight_decay"] = (5e-4, 10 ** random_state.uniform(-6, -2))
    
    hparams["adam_beta1"]=(0.9, 10 ** random_state.uniform(-6, -2))#        hparams["beta1"] = (0.5, random_state.choice([0.0, 0.5]))
    hparams["adam_beta2"]=(0.999, 10 ** random_state.uniform(-6, -2))
    hparams["sgd_momentum"]=(0.9, 10 ** random_state.uniform(-6, -2))#        hparams["beta1"] = (0.5, random_state.choice([0.0, 0.5]))
    hparams["sgd_dampening"]=(0, 10 ** random_state.uniform(-6, -2))
    hparams["sgd_nesterov"]=( False,False)

    hparams["rmsprop_alpha"]=(0.99, 10 ** random_state.uniform(-6, -2))

    hparams['nonlinear_classifier']=( False,False)


    hparams["freeze_bn"] = (True, True)
    hparams["pretrained"] = (True, True)  # only for ResNet

    hparams["w_adv"] = (0.1, 10 ** random_state.uniform(-2, 1))

    return hparams


def default_hparams(algorithm, dataset):
    dummy_random_state = np.random.RandomState(0)
    return {a: b for a, (b, c) in _hparams(algorithm, dataset, dummy_random_state).items()}


def random_hparams(algorithm, dataset, seed):
    random_state = np.random.RandomState(seed)
    return {a: c for a, (b, c) in _hparams(algorithm, dataset, random_state).items()}
