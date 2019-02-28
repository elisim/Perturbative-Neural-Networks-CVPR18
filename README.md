# Perturbative Neural Networks
Perturbative Neural Networks implementation in PyTorch by Eli Simhayev & Tal Yitzhak.

## Getting Started
Run code using the command:
```
python main.py --net-type <net-type> --dataset_train <train> --dataset_test <test>
```

### where:
**net-type** - lenet or pertube_resnet18 

**dataset_train/ dataset_test** - MNIST, CFIAR-10, EMNIST, CIFAR-100

### Full descriptipon for all the arguments could be fonund using to command: 

```
python main.py --help
```
### Gives:

```
usage: main.py [-h] [--dataset-test] [--dataset-train] [--dataroot] [--save]
               [--logs] [--resume] [--transfer] [--use_act | --no-use_act]
               [--unique_masks | --no-unique_masks] [--debug | --no-debug]
               [--train_masks | --no-train_masks] [--mix_maps | --no-mix_maps]
               [--filter_size] [--first_filter_size] [--nfilters] [--nmasks]
               [--level] [--scale_noise] [--noise_type] [--dropout]
               [--net-type] [--act] [--pool_type] [--batch-size] [--nepochs]
               [--nthreads] [--manual-seed] [--optim-method] [--learning-rate]
               [--momentum] [--weight-decay] [--adam-beta1] [--adam-beta2]

PNN

optional arguments:
  -h, --help            show this help message and exit
  --dataset-test        name of testing dataset
  --dataset-train       name of training dataset
  --dataroot            path to the data
  --save                save the trained models here
  --logs                save the training log files here
  --resume              full path of models to resume training
  --transfer            use transfer learning or not
  --use_act
  --no-use_act
  --unique_masks
  --no-unique_masks
  --debug
  --no-debug
  --train_masks
  --no-train_masks
  --mix_maps
  --no-mix_maps
  --filter_size         use conv layer with this kernel size in FirstLayer
  --first_filter_size   use conv layer with this kernel size in FirstLayer
  --nfilters            number of filters in each layer
  --nmasks              number of noise masks per input channel (fan out)
  --level               noise level for uniform noise
  --scale_noise         noise level for uniform noise
  --noise_type          type of noise
  --dropout             dropout parameter
  --net-type            type of network
  --act                 activation function (for both perturb and conv layers)
  --pool_type           pooling function (max or avg)
  --batch-size          batch size for training
  --nepochs             number of epochs to train
  --nthreads            number of threads for data loading
  --manual-seed         manual seed for randomness
  --optim-method        the optimization routine
  --learning-rate       learning rate
  --momentum            momentum
  --weight-decay        weight decay
  --adam-beta1          Beta 1 parameter for Adam
  --adam-beta2          Beta 2 parameter for Adam
```

### Prerequisites
```
Python 3.7.1 
PyTorch >= 1.0.0 
```

## References
* PNN Project Page: [Perturbative Neural Networks (PNN)](http://xujuefei.com/pnn.html)

* [Felix Juefei-Xu](http://xujuefei.com), [Vishnu Naresh Boddeti](http://vishnu.boddeti.net/), and Marios Savvides, [**Perturbative Neural Networks**](https://arxiv.org/pdf/1806.01817v1.pdf), in *Proceedings of the IEEE Computer Vision and Pattern Recognition (CVPR), 2018*.
