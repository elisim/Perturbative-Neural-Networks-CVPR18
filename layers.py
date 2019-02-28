import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import act_fn

# define device to cuda if exist
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


class PerturbLayerFirst(nn.Module):
    """
    Treat the first-layer noise module differently from the noise modules in the rest of the layers,
    because in the CVPR version of PNN, the first layer uses 3x3 or 7x7 spatial convolution
    as feature extraction. All subsequent layers use the perturbation noise modules as described in the paper.
    (duplicate class of the perturbation layer)
    """
    def __init__(self,
                 in_channels=None,
                 out_channels=None,
                 nmasks=None,
                 level=None,
                 filter_size=None,
                 debug=False,
                 use_act=False,
                 act=None,
                 stride=1,
                 unique_masks=False,
                 mix_maps=None,
                 train_masks=False,
                 noise_type='uniform',
                 input_size=None):
        """
        :param nmasks: number of perturbation masks per input channel
        :param level:  noise magnitude
        :param filter_size: if filter_size=0, layers=(perturb, conv_1x1) else layers=(conv_NxN), N=filter_size
        :param debug: debug mode or not
        :param use_act: whether to use activation immediately after perturbing input (set it to False for the first layer)
        :param stride: stride
        :param unique_masks: same set or different sets of nmasks per input channel
        :param mix_maps: whether to apply second 1x1 convolution after perturbation, to mix output feature maps
        :param train_masks: whether to treat noise masks as regular trainable parameters of the model
        :param noise_type: normal or uniform
        :param input_size: input image resolution (28 for MNIST, 32 for CIFAR), needed to construct masks
        """
        super(PerturbLayerFirst, self).__init__()
        self.nmasks = nmasks
        self.unique_masks = unique_masks
        self.train_masks = train_masks
        self.level = level
        self.filter_size = filter_size
        self.use_act = use_act
        self.act = act_fn('sigmoid')
        self.debug = debug
        self.noise_type = noise_type
        self.in_channels = in_channels
        self.input_size = input_size
        self.mix_maps = mix_maps

        if filter_size == 1:
            padding = 0
            bias = True
        elif filter_size == 3 or filter_size == 5:
            padding = 1
            bias = False
        elif filter_size == 7:
            stride = 2
            padding = 3
            bias = False

        if self.filter_size > 0:
            self.noise = None
            self.layers = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=filter_size, padding=padding, stride=stride, bias=bias),
                nn.BatchNorm2d(out_channels),
                self.act
            )


        else: # layers=(perturb, conv_1x1)
            noise_channels = in_channels if self.unique_masks else 1
            shape = (1, noise_channels, self.nmasks, input_size, input_size)

            self.noise = nn.Parameter(torch.Tensor(*shape), requires_grad=self.train_masks)
            if noise_type == "uniform":
                self.noise.data.uniform_(-1, 1)
            elif self.noise_type == 'normal':
                self.noise.data.normal_()
            else:
                print('\n\nNoise type {} is not supported / understood\n\n'.format(self.noise_type))

            if nmasks != 1:
                if out_channels % in_channels != 0:
                    print('\n\n\nnfilters must be divisible by 3 if using multiple noise masks per input channel\n\n\n')
                groups = in_channels
            else:
                groups = 1

            self.layers = nn.Sequential(
                                    nn.BatchNorm2d(in_channels*self.nmasks),
                                    self.act,
                                    nn.Conv2d(in_channels*self.nmasks, out_channels, kernel_size=1, stride=1, groups=groups),
                                    nn.BatchNorm2d(out_channels),
                                    self.act,
                                    )
            if self.mix_maps:
                self.mix_layers = nn.Sequential(
                    nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=1, groups=1),
                    nn.BatchNorm2d(out_channels),
                    self.act,
                )

    def forward(self, x):
        if self.filter_size > 0:
            return self.layers(x)  #image, conv, batchnorm, relu
        else:
            y = torch.add(x.unsqueeze(2), self.noise * self.level)
            # (10, 3, 1, 32, 32) + (1, 3, 128, 32, 32) --> (10, 3, 128, 32, 32)

            y = y.view(-1, self.in_channels * self.nmasks, self.input_size, self.input_size)
            y = self.layers(y)

            if self.mix_maps:
                y = self.mix_layers(y)

            return y  #image, perturb, (relu?), conv1x1, batchnorm, relu + mix_maps (conv1x1, batchnorm relu)


class PerturbLayer(nn.Module):
    """
    Perturbation layer, as described in the paper.
    """
    def __init__(self,
                 in_channels=None,
                 out_channels=None,
                 nmasks=None,
                 level=None,
                 filter_size=None,
                 debug=False,
                 use_act=False,
                 stride=1,
                 act=None,
                 unique_masks=False,
                 mix_maps=None,
                 train_masks=False,
                 noise_type='uniform',
                 input_size=None):
        super(PerturbLayer, self).__init__()
        self.nmasks = nmasks
        self.unique_masks = unique_masks
        self.train_masks = train_masks
        self.level = level
        self.filter_size = filter_size
        self.use_act = use_act
        self.act = act_fn(act)
        self.debug = debug
        self.noise_type = noise_type
        self.in_channels = in_channels
        self.input_size = input_size
        self.mix_maps = mix_maps

        if filter_size == 1:
            padding = 0
            bias = True
        elif filter_size == 3 or filter_size == 5:
            padding = 1
            bias = False
        elif filter_size == 7:
            stride = 2
            padding = 3
            bias = False

        if self.filter_size > 0:
            self.noise = None
            self.layers = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=filter_size, padding=padding, stride=stride, bias=bias),
                nn.BatchNorm2d(out_channels),
                self.act
            )
        else:
            noise_channels = in_channels if self.unique_masks else 1
            shape = (1, noise_channels, self.nmasks, input_size, input_size)  # can't dynamically reshape masks in forward if we want to train them
            self.noise = nn.Parameter(torch.Tensor(*shape), requires_grad=self.train_masks)
            if noise_type == "uniform":
                self.noise.data.uniform_(-1, 1)
            elif self.noise_type == 'normal':
                self.noise.data.normal_()
            else:
                print('\n\nNoise type {} is not supported / understood\n\n'.format(self.noise_type))

            if nmasks != 1:
                if out_channels % in_channels != 0:
                    print('\n\n\nnfilters must be divisible by 3 if using multiple noise masks per input channel\n\n\n')
                groups = in_channels
            else:
                groups = 1

            self.layers = nn.Sequential(
                nn.Conv2d(in_channels*self.nmasks, out_channels, kernel_size=1, stride=1, groups=groups),
                nn.BatchNorm2d(out_channels),
                self.act,
            )
            if self.mix_maps:
                self.mix_layers = nn.Sequential(
                    nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=1, groups=1),
                    nn.BatchNorm2d(out_channels),
                    self.act,
                )

    def forward(self, x):
        if self.filter_size > 0:
            return self.layers(x)  #image, conv, batchnorm, relu
        else:
            y = torch.add(x.unsqueeze(2), self.noise * self.level)  # (10, 3, 1, 32, 32) + (1, 3, 128, 32, 32) --> (10, 3, 128, 32, 32)

            if self.use_act:
                y = self.act(y)

            y = y.view(-1, self.in_channels * self.nmasks, self.input_size, self.input_size)
            y = self.layers(y)

            if self.mix_maps:
                y = self.mix_layers(y)

            return y  #image, perturb, (relu?), conv1x1, batchnorm, relu + mix_maps (conv1x1, batchnorm relu)


class PerturbBasicBlock(nn.Module):
    """
    Two PerturbLayer-s with pooling between
    Pool type can be assigned by 'pool_type'
    """
    expansion = 1

    def __init__(self,
                 in_channels=None,
                 out_channels=None,
                 stride=1,
                 shortcut=None,
                 nmasks=None,
                 train_masks=False,
                 level=None,
                 use_act=False,
                 filter_size=None,
                 act=None,
                 unique_masks=False,
                 noise_type=None,
                 input_size=None,
                 pool_type=None,
                 mix_maps=None):
        super(PerturbBasicBlock, self).__init__()
        self.shortcut = shortcut
        if pool_type == 'max':
            pool = nn.MaxPool2d
        elif pool_type == 'avg':
            pool = nn.AvgPool2d
        else:
            raise ValueError('Pool Type {} is not supported/understood'.format(pool_type))
        self.layers = nn.Sequential(
            PerturbLayer(in_channels=in_channels,
                         out_channels=out_channels,
                         nmasks=nmasks,
                         input_size=input_size,
                         level=level,
                         filter_size=filter_size,
                         use_act=use_act,
                         train_masks=train_masks,
                         act=act,
                         unique_masks=unique_masks,
                         noise_type=noise_type,
                         mix_maps=mix_maps),
            pool(stride, stride),
            PerturbLayer(in_channels=out_channels,
                         out_channels=out_channels,
                         nmasks=nmasks,
                         input_size=input_size//stride,
                         level=level,
                         filter_size=filter_size,
                         use_act=use_act,
                         train_masks=train_masks,
                         act=act,
                         unique_masks=unique_masks,
                         noise_type=noise_type,
                         mix_maps=mix_maps),
        )

    def forward(self, x):
        residual = x
        y = self.layers(x)
        if self.shortcut:
            residual = self.shortcut(x)
        y += residual
        y = F.relu(y)
        return y


