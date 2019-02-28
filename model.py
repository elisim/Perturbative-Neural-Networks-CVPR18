import torch
import utils
import numpy as np
from torch import nn
import models


class Model:
    """
    Interface for PNN model from models.py
    """
    def __init__(self, args):
        self.cuda = torch.cuda.is_available()
        self.lr = args.learning_rate
        self.dataset_train_name = args.dataset_train
        self.nfilters = args.nfilters
        self.batch_size = args.batch_size
        self.level = args.level
        self.net_type = args.net_type
        self.nmasks = args.nmasks
        self.unique_masks = args.unique_masks
        self.filter_size = args.filter_size
        self.first_filter_size = args.first_filter_size
        self.scale_noise = args.scale_noise
        self.noise_type = args.noise_type
        self.act = args.act
        self.use_act = args.use_act
        self.dropout = args.dropout
        self.train_masks = args.train_masks
        self.debug = args.debug
        self.pool_type = args.pool_type
        self.mix_maps = args.mix_maps

        ds_config = utils.get_dataset_config(self.dataset_train_name, self.filter_size)
        self.input_size = ds_config['input_size']
        self.nclasses = ds_config['nclasses']
        self.avgpool = ds_config['avgpool']

        # init model
        self.model = getattr(models, self.net_type)(
            nfilters=self.nfilters,
            avgpool=self.avgpool,
            nclasses=self.nclasses,
            nmasks=self.nmasks,
            unique_masks=self.unique_masks,
            level=self.level,
            filter_size=self.filter_size,
            first_filter_size=self.first_filter_size,
            act=self.act,
            scale_noise=self.scale_noise,
            noise_type=self.noise_type,
            use_act=self.use_act,
            dropout=self.dropout,
            train_masks=self.train_masks,
            pool_type=self.pool_type,
            debug=self.debug,
            input_size=self.input_size,
            mix_maps=self.mix_maps
        )

        self.loss_fn = nn.CrossEntropyLoss()

        # move all params to GPU
        if self.cuda:
            self.model = self.model.cuda()
            self.loss_fn = self.loss_fn.cuda()

        self.optimizer = utils.get_optimizer(self.model, args, self.lr)

    def train(self, epoch, dataloader):
        self.model.train()

        lr = utils.learning_rate_scheduler(self.dataset_train_name, epoch+1, self.lr)

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

        losses = []
        accuracies = []
        for i, (input, label) in enumerate(dataloader):
            if self.cuda:
                label = label.cuda()
                input = input.cuda()

            output = self.model(input)
            loss = self.loss_fn(output, label)
            if self.debug:
                print('\nBatch:', i)
            # reference: https://discuss.pytorch.org/t/how-are-optimizer-step-and-loss-backward-related/7350
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            pred = output.data.max(1)[1]

            acc = pred.eq(label.data).cpu().sum()*100.0 / self.batch_size

            losses.append(loss.item())
            accuracies.append(acc)

        return np.mean(losses), np.mean(accuracies)

    def test(self, dataloader):
        self.model.eval()
        losses = []
        accuracies = []
        with torch.no_grad():
            for i, (input, label) in enumerate(dataloader):
                if self.cuda:
                    label = label.cuda()
                    input = input.cuda()

                output = self.model(input)
                loss = self.loss_fn(output, label)

                pred = output.data.max(1)[1]
                acc = pred.eq(label.data).cpu().sum()*100.0 / self.batch_size
                losses.append(loss.item())
                accuracies.append(acc)

        return np.mean(losses), np.mean(accuracies)
