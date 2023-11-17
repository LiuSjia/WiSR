import torch


def get_optimizer(hparams, params):
    name = hparams["optimizer"].lower()


    if name in ["adam","adamw","radam"]:
        optimizers = {"adam": torch.optim.Adam, "adamw": torch.optim.AdamW}
        optim_cls = optimizers[name]
        optimizer =optim_cls(
            params,
            lr=hparams["lr"],
            weight_decay=hparams["weight_decay"],
            betas=(hparams["adam_beta1"], hparams["adam_beta2"])
        )
    elif name == "amsgrad":
        optimizer =torch.optim.Adam(
            params,
            lr=hparams["lr"],
            weight_decay=hparams["weight_decay"],
            betas=(hparams["adam_beta1"], hparams["adam_beta2"]),
            amsgrad=True,
        )
    elif name=="sgd": 
        optimizer = torch.optim.SGD(
                params,
                lr=hparams["lr"],
                momentum=0.9,
                weight_decay=hparams["weight_decay"],
                dampening=0,
                nesterov=False,
            )
    elif name=="rmsprop": 
        optimizer = torch.optim.RMSprop(
                params,
                lr=hparams["lr"],
                momentum=hparams["momentum"],
                weight_decay=hparams["weight_decay"],
                alpha=hparams["rmsprop_alpha"],
            )
    else:
        raise NotImplementedError(f"Optimizer {name} not implemented yet!")

    return optimizer
