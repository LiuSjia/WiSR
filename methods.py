# Copyright (c) Kakao Brain. All Rights Reserved.
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from optimizers import get_optimizer
from backbones import networks

class Algorithm(torch.nn.Module):
    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(Algorithm, self).__init__()
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.num_domains = num_domains
        self.hparams = hparams

    def update(self, minibatches, unlabeled=None):
        """
        Perform one update step, given a list of (x, y) tuples for all
        environments.

        Admits an optional list of unlabeled minibatches from the test domains,
        when task is domain_adaptation.
        """
        raise NotImplementedError

    def predict(self, x):
        raise NotImplementedError

    def forward(self, x):
        return self.predict(x)

    def new_optimizer(self, parameters):
        # if name is None:
        #     name=self.hparams["optimizer"]
        optimizer = get_optimizer(self.hparams,parameters)
            
        return optimizer

    def clone(self):
        clone = copy.deepcopy(self)
        clone.optimizer = self.new_optimizer(clone.network.parameters())
        clone.optimizer.load_state_dict(self.optimizer.state_dict())

        return clone


class ERM(Algorithm):
    """
    Empirical Risk Minimization (ERM)
    """

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(ERM, self).__init__(input_shape, num_classes, num_domains, hparams)
        self.featurizer = networks.Featurizer(input_shape, self.hparams)
        self.classifier = nn.Linear(self.featurizer.n_outputs, num_classes)
        self.network = nn.Sequential(self.featurizer, self.classifier)
        self.optimizer = self.new_optimizer(self.network.parameters())

    def update(self, minibatches, unlabeled=None):
        all_x = torch.cat([x for x,y in minibatches])
        all_y = torch.cat([y for x,y in minibatches])
        loss = F.cross_entropy(self.predict(all_x), all_y)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {"loss": loss.item()}

    def predict(self, x):
        return self.network(x)
    
class WiSR(Algorithm):
    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(WiSR, self).__init__(input_shape, num_classes, num_domains, hparams)
        
        print("Building F")
        self.featurizer = networks.Featurizer(input_shape, self.hparams)
        # gesture network
        self.classifier_g = nn.Linear(self.featurizer.n_outputs, num_classes)
        # style network
        self.classifier_s = nn.Linear(self.featurizer.n_outputs, num_domains)
        #self.classifier_s = networks.MLP(self.featurizer.n_outputs, num_domains, self.hparams)
        
        
        self.optimizer_f = self.new_optimizer(self.featurizer.parameters())
        self.optimizer_g = self.new_optimizer(self.classifier_g.parameters())
        self.optimizer_s = self.new_optimizer(self.classifier_s.parameters())
        self.weight_adv = hparams["w_adv"]

    def forward_g(self, x):
        # learning gesture network on randomized style
        return self.classifier_g(self.randomize(self.featurizer(x), "style"))

    def forward_s(self, x):
        # learning style network on randomized gesture
        return self.classifier_s(self.randomize(self.featurizer(x), "gesture"))

    def randomize(self, x, what="style", eps=1e-5):#torch.Size([128, 512])
        sizes = x.size()
        alpha = torch.rand(sizes[0],1).cuda()
        if len(sizes) == 4:
            x = x.view(sizes[0], sizes[1], -1)
            alpha = alpha.unsqueeze(-1)

        mean = x.mean(-1, keepdim=True)
        var = x.var(-1, keepdim=True)

        

        idx_swap = torch.randperm(sizes[0])
        if what == "style":
            mean = alpha * mean + (1 - alpha) * mean[idx_swap]
            var = alpha * var + (1 - alpha) * var[idx_swap]
            
            x = (x - mean) / (var + eps).sqrt()
            x = x * (var + eps).sqrt() + mean
        else:
            x = alpha * x + (1 - alpha) * x[idx_swap]
            
            x = (x - mean) / (var + eps).sqrt()
            x = x * (var + eps).sqrt() + mean

        
        return x.view(*sizes)

    def update(self, minibatches, unlabeled=None):
        all_x = torch.cat([x for x, y in minibatches])
        all_y = torch.cat([y for x, y in minibatches])
        all_d = torch.cat([
            torch.full((x.shape[0],), i, dtype=torch.int64, device="cuda")
            for i, (x, y) in enumerate(minibatches)
        ])

        # learn gesture feature
        self.optimizer_f.zero_grad()
        self.optimizer_g.zero_grad()
        loss_g = F.cross_entropy(self.forward_g(all_x), all_y)
        loss_g.backward()
        self.optimizer_f.step()
        self.optimizer_g.step()

        # learn style
        self.optimizer_s.zero_grad()
        loss_s = F.cross_entropy(self.forward_s(all_x), all_d)
        loss_s.backward()
        self.optimizer_s.step()

        # learn adversary
        self.optimizer_f.zero_grad()
        loss_adv = -F.log_softmax(self.forward_s(all_x), dim=1).mean(1).mean()
        loss_adv = loss_adv * self.weight_adv
        loss_adv.backward()
        self.optimizer_f.step()

        return {
            "loss_g": loss_g.item(),
            "loss_s": loss_s.item(),
            "loss_adv": loss_adv.item(),
        }

    def predict(self, x):
        return self.classifier_g(self.featurizer(x))  
    
