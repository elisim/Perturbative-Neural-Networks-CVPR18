from layers import *

# define device to cuda if exist
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


class PerturbResNet(nn.Module):
    """
    ResNet-18 architecture where each convolution is replaced with perturbation.
    The implementation motivated by https://github.com/kuangliu/pytorch-cifar/blob/master/models/resnet.py
    """
    def __init__(self,
                 block,
                 nblocks=None,
                 avgpool=None,
                 nfilters=None,
                 nclasses=None,
                 nmasks=None,
                 input_size=32,
                 level=None,
                 filter_size=None,
                 first_filter_size=None,
                 use_act=False,
                 train_masks=False,
                 mix_maps=None,
                 act=None,
                 scale_noise=1,
                 unique_masks=False,
                 debug=False,
                 noise_type=None,
                 pool_type=None):
        super(PerturbResNet, self).__init__()
        self.nfilters = nfilters
        self.unique_masks = unique_masks
        self.noise_type = noise_type
        self.train_masks = train_masks
        self.pool_type = pool_type
        self.mix_maps = mix_maps
        self.act = act_fn(act)

        layers = [PerturbLayerFirst(in_channels=3, out_channels=3*nfilters, nmasks=nfilters*5, level=level*scale_noise*20,
                debug=debug, filter_size=first_filter_size, use_act=use_act, train_masks=train_masks, input_size=input_size,
                act=act, unique_masks=self.unique_masks, noise_type=self.noise_type, mix_maps=mix_maps)] # scale noise 20x at 1st layer

        if first_filter_size == 7:
            layers.append(nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

        self.pre_layers = nn.Sequential(*layers,
                          nn.Conv2d(self.nfilters*3*1, self.nfilters, kernel_size=1, stride=1, bias=False), # mapping 10*nfilters back to nfilters with 1x1 conv
                          nn.BatchNorm2d(self.nfilters),
                          self.act
                          )

        self.layer1 = self._make_layer(block, 1*nfilters, nblocks[0], stride=1, level=level, nmasks=nmasks, use_act=True,
                                            filter_size=filter_size, act=act, input_size=input_size)
        self.layer2 = self._make_layer(block, 2*nfilters, nblocks[1], stride=2, level=level, nmasks=nmasks, use_act=True,
                                            filter_size=filter_size, act=act, input_size=input_size)
        self.layer3 = self._make_layer(block, 4*nfilters, nblocks[2], stride=2, level=level, nmasks=nmasks, use_act=True,
                                            filter_size=filter_size, act=act, input_size=input_size//2)
        self.layer4 = self._make_layer(block, 8*nfilters, nblocks[3], stride=2, level=level, nmasks=nmasks, use_act=True,
                                            filter_size=filter_size, act=act, input_size=input_size//4)
        self.avgpool = nn.AvgPool2d(avgpool, stride=1)
        self.linear = nn.Linear(8*nfilters*block.expansion, nclasses)

    def _make_layer(self, block, out_channels, nblocks, stride=1, level=0.2, nmasks=None, use_act=False,
                                            filter_size=None, act=None, input_size=None):
        shortcut = None
        if stride != 1 or self.nfilters != out_channels * block.expansion:
            shortcut = nn.Sequential(
                nn.Conv2d(self.nfilters, out_channels * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * block.expansion),
            )
        layers = []
        layers.append(block(self.nfilters, out_channels, stride, shortcut, level=level, nmasks=nmasks, use_act=use_act,
                        filter_size=filter_size, act=act, unique_masks=self.unique_masks, noise_type=self.noise_type,
                        train_masks=self.train_masks, input_size=input_size, pool_type=self.pool_type, mix_maps=self.mix_maps))
        self.nfilters = out_channels * block.expansion
        for i in range(1, nblocks):
            layers.append(block(self.nfilters, out_channels, level=level, nmasks=nmasks, use_act=use_act,
                            train_masks=self.train_masks, filter_size=filter_size, act=act, unique_masks=self.unique_masks,
                            noise_type=self.noise_type, input_size=input_size//stride, pool_type=self.pool_type, mix_maps=self.mix_maps))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.pre_layers(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.linear(x)
        return x


class LeNet(nn.Module):
    """
    LeNet architecture where each convolution is replaced with perturbation.
    The implementation motivated by https://github.com/kuangliu/pytorch-cifar/blob/master/models/lenet.py
    """
    def __init__(self,
                 nfilters=None,
                 nclasses=None,
                 nmasks=None,
                 level=None,
                 filter_size=None,
                 linear=128,
                 input_size=28,
                 debug=False,
                 scale_noise=1,
                 act='relu',
                 use_act=False,
                 first_filter_size=None,
                 pool_type=None,
                 dropout=None,
                 unique_masks=False,
                 train_masks=False,
                 noise_type='uniform',
                 mix_maps=None):
        super(LeNet, self).__init__()
        if filter_size == 5:
            n = 5
        else:
            n = 4

        if input_size == 32:
            first_channels = 3
        elif input_size == 28:
            first_channels = 1

        if pool_type == 'max':
            pool = nn.MaxPool2d
        elif pool_type == 'avg':
            pool = nn.AvgPool2d
        else:
            raise ValueError('Pool Type {} is not supported'.format(pool_type))

        self.linear1 = nn.Linear(nfilters*n*n, linear)
        self.linear2 = nn.Linear(linear, nclasses)
        self.dropout = nn.Dropout(p=dropout)
        self.act = act_fn(act)
        self.batch_norm = nn.BatchNorm1d(linear)

        self.first_layers = nn.Sequential(
            PerturbLayer(in_channels=first_channels, out_channels=nfilters, nmasks=nmasks, level=level*scale_noise,
                         filter_size=first_filter_size, use_act=use_act, act=act, unique_masks=unique_masks,
                         train_masks=train_masks, noise_type=noise_type, input_size=input_size, mix_maps=mix_maps),
            pool(kernel_size=3, stride=2, padding=1),

            PerturbLayer(in_channels=nfilters, out_channels=nfilters, nmasks=nmasks, level=level, filter_size=filter_size,
                         use_act=True, act=act, unique_masks=unique_masks, debug=debug, train_masks=train_masks,
                         noise_type=noise_type, input_size=input_size//2, mix_maps=mix_maps),
            pool(kernel_size=3, stride=2, padding=1),

            PerturbLayer(in_channels=nfilters, out_channels=nfilters, nmasks=nmasks, level=level, filter_size=filter_size,
                         use_act=True, act=act, unique_masks=unique_masks, train_masks=train_masks, noise_type=noise_type,
                         input_size=input_size//4, mix_maps=mix_maps),
            pool(kernel_size=3, stride=2, padding=1),
        )

        self.last_layers = nn.Sequential(
            self.dropout,
            self.linear1,
            self.batch_norm,
            self.act,
            self.dropout,
            self.linear2,
        )

    def forward(self, x):
        x = self.first_layers(x)
        x = x.view(x.size(0), -1)
        x = self.last_layers(x)
        return x


def perturb_resnet18(nfilters, avgpool=4, nclasses=10, nmasks=32, level=0.1, filter_size=0, first_filter_size=0,
                     pool_type=None, input_size=None, scale_noise=1, act='relu', use_act=True, dropout=0.5,
                     unique_masks=False, debug=False, noise_type='uniform', train_masks=False, mix_maps=None):
    return PerturbResNet(PerturbBasicBlock, [2, 2, 2, 2], nfilters=nfilters, avgpool=avgpool, nclasses=nclasses, pool_type=pool_type,
                         scale_noise=scale_noise, nmasks=nmasks, level=level, filter_size=filter_size, train_masks=train_masks,
                         first_filter_size=first_filter_size, act=act, use_act=use_act, unique_masks=unique_masks,
                         debug=debug, noise_type=noise_type, input_size=input_size, mix_maps=mix_maps)


def lenet(nfilters, avgpool=None, nclasses=10, nmasks=32, level=0.1, filter_size=3, first_filter_size=0,
          pool_type=None, input_size=None, scale_noise=1, act='relu', use_act=True, dropout=0.5,
          unique_masks=False, debug=False, noise_type='uniform', train_masks=False, mix_maps=None):
    return LeNet(nfilters=nfilters, nclasses=nclasses, nmasks=nmasks, level=level, filter_size=filter_size, pool_type=pool_type,
                 scale_noise=scale_noise, act=act, first_filter_size=first_filter_size, input_size=input_size, mix_maps=mix_maps,
                 use_act=use_act, dropout=dropout, unique_masks=unique_masks, debug=debug, noise_type=noise_type, train_masks=train_masks)
