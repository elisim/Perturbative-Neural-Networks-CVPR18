import torch
import random
from dataloader import Dataloader
import utils
import os
from datetime import datetime
import argparse
import transfer
import warnings
from model import Model


result_path = "results/"
result_path = os.path.join(result_path, datetime.now().strftime('%Y-%m-%d_%H-%M-%S/'))

parser = argparse.ArgumentParser(description='PNN')
warnings.filterwarnings("ignore")

# ======================== Data Setings ============================================
parser.add_argument('--dataset-test', type=str, default='CIFAR10', metavar='', help='name of testing dataset')
parser.add_argument('--dataset-train', type=str, default='CIFAR10', metavar='', help='name of training dataset')
parser.add_argument('--dataroot', type=str, default='./data', metavar='', help='path to the data')
parser.add_argument('--save', type=str, default=result_path +'Save', metavar='', help='save the trained models here')
parser.add_argument('--logs', type=str, default=result_path +'Logs', metavar='', help='save the training log files here')
parser.add_argument('--resume', type=str, default=None, metavar='', help='full path of models to resume training')
parser.add_argument('--transfer', type=bool, default=False, metavar='', help='use transfer learning or not')

# ======================== Network Model Setings ===================================

feature_parser = parser.add_mutually_exclusive_group(required=False)
feature_parser.add_argument('--use_act', dest='use_act', action='store_true')
feature_parser.add_argument('--no-use_act', dest='use_act', action='store_false')
parser.set_defaults(use_act=False)

feature_parser = parser.add_mutually_exclusive_group(required=False)
feature_parser.add_argument('--unique_masks', dest='unique_masks', action='store_true')
feature_parser.add_argument('--no-unique_masks', dest='unique_masks', action='store_false')
parser.set_defaults(unique_masks=True)

feature_parser = parser.add_mutually_exclusive_group(required=False)
feature_parser.add_argument('--debug', dest='debug', action='store_true')
feature_parser.add_argument('--no-debug', dest='debug', action='store_false')
parser.set_defaults(debug=False)

feature_parser = parser.add_mutually_exclusive_group(required=False)
feature_parser.add_argument('--train_masks', dest='train_masks', action='store_true')
feature_parser.add_argument('--no-train_masks', dest='train_masks', action='store_false')
parser.set_defaults(train_masks=False)

feature_parser = parser.add_mutually_exclusive_group(required=False)
feature_parser.add_argument('--mix_maps', dest='mix_maps', action='store_true')
feature_parser.add_argument('--no-mix_maps', dest='mix_maps', action='store_false')
parser.set_defaults(mix_maps=False)

parser.add_argument('--filter_size', type=int, default=0, metavar='', help='use conv layer with this kernel size in FirstLayer')
parser.add_argument('--first_filter_size', type=int, default=0, metavar='', help='use conv layer with this kernel size in FirstLayer')
parser.add_argument('--nfilters', type=int, default=64, metavar='', help='number of filters in each layer')
parser.add_argument('--nmasks', type=int, default=1, metavar='', help='number of noise masks per input channel (fan out)')
parser.add_argument('--level', type=float, default=0.1, metavar='', help='noise level for uniform noise')
parser.add_argument('--scale_noise', type=float, default=1.0, metavar='', help='noise level for uniform noise')
parser.add_argument('--noise_type', type=str, default='uniform', metavar='', help='type of noise')
parser.add_argument('--dropout', type=float, default=1e-4, metavar='', help='dropout parameter')
parser.add_argument('--net-type', type=str, default='resnet18', metavar='', help='type of network')
parser.add_argument('--act', type=str, default='relu', metavar='', help='activation function (for both perturb and conv layers)')
parser.add_argument('--pool_type', type=str, default='max', metavar='', help='pooling function (max or avg)')

# ======================== Training Settings =======================================
parser.add_argument('--batch-size', type=int, default=64, metavar='', help='batch size for training')
parser.add_argument('--nepochs', type=int, default=150, metavar='', help='number of epochs to train')
parser.add_argument('--nthreads', type=int, default=4, metavar='', help='number of threads for data loading')
parser.add_argument('--manual-seed', type=int, default=1, metavar='', help='manual seed for randomness')

# ======================== Hyperparameter Setings ==================================
parser.add_argument('--optim-method', type=str, default='Adam', metavar='', help='the optimization routine ')
parser.add_argument('--learning-rate', type=float, default=1e-4, metavar='', help='learning rate')
parser.add_argument('--momentum', type=float, default=0.9, metavar='', help='momentum')
parser.add_argument('--weight-decay', type=float, default=1e-4, metavar='', help='weight decay')
parser.add_argument('--adam-beta1', type=float, default=0.9, metavar='', help='Beta 1 parameter for Adam')
parser.add_argument('--adam-beta2', type=float, default=0.999, metavar='', help='Beta 2 parameter for Adam')

args = parser.parse_args()
random.seed(args.manual_seed)
torch.manual_seed(args.manual_seed)
utils.save_args(args)

print('\n\n****** Creating {} model ******'.format(args.net_type))
setup = Model(args)
print("model created successfully!")
print('\n\n****** Preparing {} dataset *******'.format(args.dataset_train))
dataloader = Dataloader(args, setup.input_size)
loader_train, loader_test = dataloader.create()
print('data prepared successfully!')

# initialize model:
if args.resume is None:
    model = setup.model
    model.apply(utils.weights_init)
    train = setup.train
    test = setup.test
    init_epoch = 0
    acc_best = 0
    best_epoch = 0
    if os.path.isdir(args.save) == False:
        os.makedirs(args.save)

else: # Transfer Learning
    print('\n\nLoading model from saved checkpoint at {}\n\n'.format(args.resume))
    setup.model = torch.load(args.resume)
    model = setup.model
    train = setup.train
    test = setup.test
    if args.transfer:
        model = transfer.transfer(model, setup.nclasses)
        init_epoch = 0
        acc_best = 0
        best_epoch = 0
        if os.path.isdir(args.save) == False:
            os.makedirs(args.save)
    else:
        te_loss, te_acc = test(loader_test)
        init_epoch = int(args.resume.split('_')[3])  # extract N from 'results/xxx_xxx/Save/model_epoch_N_acc_nn.nn.pth'
        print('\n\nRestored Model Accuracy (epoch {:d}): {:.2f}\n\n'.format(init_epoch, te_acc))
        acc_best = te_acc
        best_epoch = init_epoch
        args.save = '/'.join(args.resume.split('/')[:-1])
        init_epoch += 1


print('\n\n****** Model Configuration ******')
for arg in vars(args):
    print(arg, getattr(args, arg))
print('\n')


accuracies = []
### Train loop
for epoch in range(init_epoch, args.nepochs, 1):

    tr_loss, tr_acc = setup.train(epoch, loader_train)
    te_loss, te_acc = setup.test(loader_test)

    accuracies.append(te_acc)

    if te_acc > acc_best:
        print('{}  Epoch {:d}/{:d}  Train: Loss {:.2f} Accuracy {:.2f} Test: Loss {:.2f} Accuracy {:.2f} (best result, saving to {})'.format(
                        str(datetime.now())[:-7], epoch, args.nepochs, tr_loss, tr_acc, te_loss, te_acc, args.save))
        model_best = True
        acc_best = te_acc
        best_epoch = epoch
        torch.save(model, args.save + '/model_epoch_{:d}_acc_{:.2f}.pth'.format(epoch, te_acc))
    else:
        if epoch == 0:
            print('\n')
        print('{}  Epoch {:d}/{:d}  Train: Loss {:.2f} Accuracy {:.2f} Test: Loss {:.2f} Accuracy {:.2f}'.format(
                                str(datetime.now())[:-7], epoch, args.nepochs, tr_loss, tr_acc, te_loss, te_acc))

print('\n\nBest Accuracy: {:.2f}  (epoch {:d})\n\n'.format(acc_best, best_epoch))
print('\n\nTest Accuracies:\n\n')

for v in accuracies:
    print('{:.2f}'.format(v)+', ', end='')