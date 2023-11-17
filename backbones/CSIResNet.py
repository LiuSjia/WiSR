import torch
import torch.nn as nn
import torch.nn.functional as F

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv1d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv1d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm1d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm1d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, planes)
        self.bn1 = nn.BatchNorm1d(planes)
        self.conv2 = conv3x3(planes, planes, stride)
        self.bn2 = nn.BatchNorm1d(planes)
        self.conv3 = conv1x1(planes, planes * self.expansion)
        self.bn3 = nn.BatchNorm1d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

    
class ResNet(nn.Module):

    def __init__(
        self, 
        block, 
        layers,  
        inchannel=90, 
        activity_num=6, 
        location_num=16,
        ms_class=None,
        ms_layers=[],
        ms_p=0.5,
        ms_a=0.1,
        if_classfier=True,
        **kwargs
        ):
        super(ResNet, self).__init__()
        self.inplanes = 128
        self.conv1 = nn.Conv1d(inchannel, self.inplanes, kernel_size=7, stride=2, padding=3,
                                 bias=False)

        self.bn1 = nn.BatchNorm1d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 128, layers[0], stride=1)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        self.if_classfier=if_classfier
        # 
        self.l1 = nn.Conv1d(512 * block.expansion, 512 * block.expansion, kernel_size=3, stride=1, padding=0, bias=False)
        self.l2 = nn.BatchNorm1d(512 * block.expansion)
        self.l3 = nn.ReLU(inplace=True)
        self.l4 = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(self.l1,self.l2,self.l3,self.l4 )
        self.act_fc = nn.Linear(512 * block.expansion, activity_num)

        self.LOCClassifier = nn.Sequential(
            nn.Conv1d(512 * block.expansion, 512 * block.expansion, kernel_size=3, stride=1, padding=0, bias=False),
            nn.BatchNorm1d(512 * block.expansion),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(1),
        )
        self.loc_fc = nn.Linear(512 * block.expansion, location_num)
        self.loc_fc_f = nn.Linear(256, location_num)
        self.out_features = 512#self.inplanes*3
        self.n_outputs = 512

        self.mixstyle = None
        if ms_layers:
            self.mixstyle = ms_class(p=ms_p, alpha=ms_a)#0.5,0.1
            for layer_name in ms_layers:
                assert layer_name in ["layer1", "layer2", "layer3"]
            print(f"Insert MixStyle after {ms_layers}")
        self.ms_layers = ms_layers

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(
                    self.inplanes, 
                    planes * block.expansion, 
                    stride
                    ),
                nn.BatchNorm1d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)


    def _upsample_add(self, x, y):
        '''Upsample and add two feature maps.
        Args:
          x: (Variable) top feature map to be upsampled.
          y: (Variable) lateral feature map.
        Returns:
          (Variable) added feature map.
        Note in PyTorch, when input size is odd, the upsampled feature map
        with `F.upsample(..., scale_factor=2, mode='nearest')`
        maybe not equal to the lateral feature map size.
        e.g.
        original input size: [N,_,15,15] ->
        conv2d feature map size: [N,_,8,8] ->
        upsampled feature map size: [N,_,16,16]
        So we choose bilinear upsample which supports arbitrary output sizes.
        '''
        _,_,L = y.size()
        return F.interpolate(x, size=L) + y


    def forward(self, x):#torch.Size([8, 90, 2500])
        x = self.conv1(x)#torch.Size([8, 128, 1250])
        x = self.bn1(x)#torch.Size([8, 128, 1250])
        x = self.relu(x)#torch.Size([8, 128, 1250])
        x = self.maxpool(x)#torch.Size([8, 128, 625])

        x = self.layer1(x)#torch.Size([8, 128, 625])
        if "layer1" in self.ms_layers:
            x = self.mixstyle(x)
        x = self.layer2(x)#torch.Size([8, 128, 313])
        if "layer2" in self.ms_layers:
            x = self.mixstyle(x)
        x = self.layer3(x)#torch.Size([8, 256, 157])
        if "layer3" in self.ms_layers:
            x = self.mixstyle(x)
        x = self.layer4(x)#torch.Size([8, 512, 79])

        if self.if_classfier:
            x = self.classifier(x)#torch.Size([8, 512, 1])
            x = x.view(x.size(0), -1)#torch.Size([8, 512])
        return x


def CSIResNet(input_shape,if_classfier=True):
    input_c=input_shape[0]
    model = ResNet(block=BasicBlock, layers=[1, 1, 1, 1],inchannel=input_c,if_classfier=if_classfier)
    # aplnet = ResNet(block=BasicBlock, layers=[2, 2, 2, 2], inchannel=52)
    # aplnet = ResNet(block=BasicBlock, layers=[3, 4, 6, 3], inchannel=52)
    #

    # aplnet = ResNet(block=Bottleneck, layers=[2, 3, 4, 6])
    return model

def get_module(module, name):
    for n, m in module.named_modules():
        if n == name:
            return m

def build_blocks(model, block_name_dict):
    #  blocks = nn.ModuleList()
    blocks = []  # saved model can be broken...
    for _key, name_list in block_name_dict.items():
        block = nn.ModuleList()
        for module_name in name_list:
            module = get_module(model, module_name)
            block.append(module)
        blocks.append(block)

    return blocks

def freeze_(model):
    """Freeze model
    Note that this function does not control BN
    """
    for p in model.parameters():
        p.requires_grad_(False)

class URCSIResNet(torch.nn.Module):
    """ResNet + FrozenBN + IntermediateFeatures
    """
    def __init__(self, input_shape, hparams, preserve_readout=False, freeze=None, feat_layers=None):
        super().__init__()
        input_c=input_shape[0]
        self.network= ResNet(block=BasicBlock, layers=[1, 1, 1, 1],inchannel=input_c)
        self.n_outputs = self.network.n_outputs
                
        block_names ={
        "stem": ["conv1", "bn1", "relu", "maxpool"],
        "block1": ["layer1"],
        "block2": ["layer2"],
        "block3": ["layer3"],
        "block4": ["layer4"],
        }

        self._features = []
        self.feat_layers = self.build_feature_hooks(feat_layers, block_names)
        self.blocks = build_blocks(self.network, block_names)

        self.freeze(freeze)

        if not preserve_readout:
            self.dropout = nn.Dropout(hparams["resnet_dropout"])
        else:
            self.dropout = nn.Identity()
            assert hparams["resnet_dropout"] == 0.0

        self.hparams = hparams
        self.freeze_bn()

    def freeze(self, freeze):
        if freeze is not None:
            if freeze == "all":
                freeze_(self.network)
            else:
                for block in self.blocks[:freeze+1]:
                    freeze_(block)

    def hook(self, module, input, output):
        self._features.append(output)

    def build_feature_hooks(self, feats, block_names):
        assert feats in ["stem_block", "block"]

        if feats is None:
            return []

        # build feat layers
        if feats.startswith("stem"):
            last_stem_name = block_names["stem"][-1]
            feat_layers = [last_stem_name]
        else:
            feat_layers = []

        for name, module_names in block_names.items():
            if name == "stem":
                continue

            module_name = module_names[-1]
            feat_layers.append(module_name)

        #  print(f"feat layers = {feat_layers}")

        for n, m in self.network.named_modules():
            if n in feat_layers:
                m.register_forward_hook(self.hook)

        return feat_layers

    def forward(self, x, ret_feats=False):
        """Encode x into a feature vector of size n_outputs."""
        self.clear_features()
        out = self.dropout(self.network(x))
        if ret_feats:
            return out, self._features
        else:
            return out

    def clear_features(self):
        self._features.clear()

    def train(self, mode=True):
        """
        Override the default train() to freeze the BN parameters
        """
        super().train(mode)
        self.freeze_bn()

    def freeze_bn(self):
        for m in self.network.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()
       