import torch
import datasets
import torchvision.transforms as transforms


class Dataloader:
    """
    Interface for load training and testing data
    """
    def __init__(self, args, input_size):
        self.args = args
        self.dataset_train_name = args.dataset_train
        self.dataset_test_name = args.dataset_test
        self.input_size = input_size

        ### Train preparation ###
        if self.dataset_train_name == 'CIFAR10' or self.dataset_train_name == 'CIFAR100':
            self.dataset_train = getattr(datasets, self.dataset_train_name)(root=self.args.dataroot, train=True, download=True,
                transform=transforms.Compose([
                    transforms.RandomCrop(self.input_size, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                    ])
                )


        elif self.dataset_train_name == 'MNIST':
            self.dataset_train = getattr(datasets, self.dataset_train_name)(root=self.args.dataroot, train=True, download=True,
                transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])
                )

        else:
            raise(Exception("Unknown Dataset"))


        ### Test preparation ###
        if self.dataset_test_name == 'CIFAR10' or self.dataset_test_name == 'CIFAR100':
            self.dataset_test = getattr(datasets, self.dataset_test_name)(root=self.args.dataroot, train=False, download=True,
                transform=transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                    ])
                )


        elif self.dataset_test_name == 'MNIST':
            self.dataset_test = getattr(datasets, self.dataset_test_name)(root=self.args.dataroot, train=False, download=True,
                transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])
                )

        else:
            raise(Exception("Unknown Dataset"))

    def create(self, flag=None):
        if flag == "Train":
            dataloader_train = torch.utils.data.DataLoader(self.dataset_train, batch_size=self.args.batch_size,
                shuffle=True, num_workers=int(self.args.nthreads), pin_memory=True)
            return dataloader_train

        if flag == "Test":
            dataloader_test = torch.utils.data.DataLoader(self.dataset_test, batch_size=self.args.batch_size,
                shuffle=False, num_workers=int(self.args.nthreads), pin_memory=True)
            return dataloader_test

        if flag == None:
            dataloader_train = torch.utils.data.DataLoader(self.dataset_train, batch_size=self.args.batch_size,
                shuffle=True, num_workers=int(self.args.nthreads), pin_memory=True)
        
            dataloader_test = torch.utils.data.DataLoader(self.dataset_test, batch_size=self.args.batch_size,
                shuffle=False, num_workers=int(self.args.nthreads), pin_memory=True)

            return dataloader_train, dataloader_test