import os
import math
import torch.optim as optim
import torch.nn as nn


def save_args(args):
    """
    :param args: program arguments
    save the programs arguments to file
    """
    path = args.logs
    if os.path.isdir(path) == False:
        os.makedirs(path)
    with open(os.path.join(path,'args.txt'), 'w') as f:
        for arg in vars(args):
            f.write(arg+' '+str(getattr(args,arg))+'\n')


def weights_init(m):
    """
    :param m: model
    init the initial weights of the model m
    """
    if isinstance(m, nn.Conv2d):
        n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        m.weight.data.normal_(0, math.sqrt(2. / n))
    elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.fill_(1)
        m.bias.data.zero_()


def act_fn(act):
    """
    :param act: activation name as string
    :return: activation as PyTorch object
    """
    acts = {
        'relu': nn.ReLU(inplace=False),
        'lrelu': nn.LeakyReLU(inplace=True),
        'prelu': nn.PReLU(),
        'rrelu': nn.RReLU(inplace=True),
        'elu': nn.ELU(inplace=True),
        'selu': nn.SELU(inplace=True),
        'tanh': nn.Tanh(),
        'sigmoid': nn.Sigmoid(),
    }
    ans = acts.get(act, None);
    if ans is None:
        raise ValueError('Activation function {} is not supported'.format(act))
    return ans


def learning_rate_scheduler(dataset, epoch, curr_lr):
    """
    :param dataset: dataset name
    :param epoch: current epoch
    :param curr_lr: current learning rate
    :return: new learning rate
    """
    if dataset == 'CIFAR10':
        new_lr = curr_lr * ((0.2 ** int(epoch >= 150)) * (0.2 ** int(epoch >= 250)) * (0.2 ** int(epoch >= 300)) * (0.2 ** int(epoch >= 350)) * (0.2 ** int(epoch >= 400)))
    elif dataset == 'CIFAR100':
        new_lr = curr_lr * ((0.1 ** int(epoch >= 80)) * (0.1 ** int(epoch >= 120)) * (0.1 ** int(epoch >= 160)))
    elif dataset == 'MNIST' or dataset == 'EMNIST':
        new_lr = curr_lr * ((0.2 ** int(epoch >= 30)) * (0.2 ** int(epoch >= 60)) * (0.2 ** int(epoch >= 90)))

    return new_lr


def get_dataset_config(dataset, filter_size):
    """
    :param dataset: dataset name
    :param filter_size: filter size for the first convolution layer
    :return: config to configure the model for the dataset (input image sizes, number of classes and avgpool param)
    """
    if dataset == "CIFAR10":
        input_size = 32
        nclasses = 10
        if filter_size < 7:
            avgpool = 4
        elif filter_size == 7:
            avgpool = 1

    elif dataset == "CIFAR100":
        input_size = 32
        nclasses = 100
        if filter_size < 7:
            avgpool = 4
        elif filter_size == 7:
            avgpool = 1

    elif dataset == "MNIST":
        nclasses = 10
        input_size = 28
        if filter_size < 7:
            avgpool = 14
        elif filter_size == 7:
            avgpool = 7

    elif dataset == "EMNIST":
        nclasses = 47
        input_size = 28
        if filter_size < 7:
            avgpool = 14
        elif filter_size == 7:
            avgpool = 7

    else:
        raise ValueError("Unknown dataset {}".format(dataset))

    return {
        'input_size': input_size,
        'nclasses': nclasses,
        'avgpool': avgpool
    }


def get_optimizer(model, args, lr):
    """
    :param model: model for the dataset
    :param args: user arguments
    :param lr: learning rate
    :return: PyTorch optimizer according to args
    """

    parameters = [p for p in model.parameters() if p.requires_grad]

    if args.optim_method == 'Adam':
       optimizer = optim.Adam(parameters, lr=lr, betas=(args.adam_beta1, args.adam_beta2), weight_decay=args.weight_decay)
    elif args.optim_method == 'RMSprop':
       optimizer = optim.RMSprop(parameters, lr=lr, momentum=args.momentum, weight_decay=args.weight_decay)
    elif args.optim_method == 'SGD':
       optimizer = optim.SGD(parameters, lr=lr, momentum=args.momentum, weight_decay=args.weight_decay, nesterov=True)
    else:
        raise ValueError(f"Unknown Optimization Method {args.optim_method}")

    return optimizer

