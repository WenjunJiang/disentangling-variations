#!/usr/bin/env python

import os
import sys
import argparse
from argparse import RawTextHelpFormatter
import time
import shutil

import matplotlib.pyplot as plt
import numpy as np
from functools import reduce
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.nn import functional as F
from torch.optim import Adam
import torch.distributed as dist
import torchvision
from torch.autograd import Variable
from torchvision.utils import save_image

from dataloader import *
from models import *
from utils import *

best_prec1 = 0
configure("logs/log-1")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

def get_attr_loss(output, attributes, flip, params):
    """
    Compute attributes loss.
    """
    assert type(flip) is bool
    k = 0
    loss = 0
    for (_, n_cat) in params.attr:
        # categorical
        x = output[:, k:k + n_cat].contiguous()
        y = attributes[:, k:k + n_cat].max(1)[1].view(-1)
        if flip:
            # generate different categories
            shift = torch.LongTensor(y.size()).random_(n_cat - 1) + 1
            y = (y + Variable(shift.cuda())) % n_cat
        loss += F.cross_entropy(x, y)
        k += n_cat
    return loss

def vae_bce_loss_function(recon_x, x, mu, logvar):
    bce = F.binary_cross_entropy(recon_x, x.view(-1, 784), size_average=False)
    # https://arxiv.org/abs/1312.6114        time1 = time.time()

    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return bce + kld

def vae_mse_loss_function(recon_x, target, mu, logvar):
    # BCE = F.binary_cross_entropy(recon_x, x.view(-1, 784), size_average=False)
    mse_loss = nn.MSELoss(size_average=False)
    MSE = mse_loss(recon_x.view(-1, 196608), target.view(-1, 196608))
    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return MSE + KLD

def main(args):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    config = NetworkConfig(args.config)
    best_prec1 = 0

    args.distributed = config.distributed['world_size'] > 1
    if args.distributed:
        print('[+] Distributed backend')
        dist.init_process_group(backend=config.distributed['dist_backend'], init_method=config.distributed['dist_url'], \
                                world_size=config.distributed['world_size'])

    # creating model instance
    model = VAE(config.hyperparameters)
    # plotting interactively
    plt.ion()
    # lat_dis = LatentDiscriminator(config.hyperparameters)
    # ptc_dis = PatchDiscriminator(config.hyperparameters)
    if args.distributed:
        model.to(device)
        model = nn.parallel.DistributedDataParallel(model)
        # lat_dis.to(device)
        # lat_dis = nn.parallel.DistributedDataParallel(lat_dis)
        # ptc_dis.to(device)
        # ptc_dis = nn.parallel.DistributedDataParallel(ptc_dis)
    elif args.gpu:
        model = nn.DataParallel(model).to(device)
        # lat_dis.to(device)
        # ptc_dis.to(device)
    else: return

    # Data Loading
    transformations = transforms.Compose([
        transforms.ToTensor()
    ])

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    images, attributes = load_celeba_images(config.data)
    train_dataset = CelebaDataset(images[0], attributes[0], config, transformations)
    valid_dataset = CelebaDataset(images[1], attributes[1], config, transformations)
    test_dataset = CelebaDataset(images[0], attributes[0], config, transformations)

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        train_sampler = None


    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=config.data['batch_size'], shuffle=config.data['shuffle'],
        num_workers=config.data['workers'], pin_memory=config.data['pin_memory'], sampler=train_sampler)

    val_loader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=config.data['batch_size'], shuffle=config.data['shuffle'],
        num_workers=config.data['workers'], pin_memory=config.data['pin_memory'])

    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=config.data['batch_size'], shuffle=config.data['shuffle'],
        num_workers=config.data['workers'], pin_memory=config.data['pin_memory'])



    trainer = Trainer('vae', config, train_loader, model)
    if args.evaluate:
        evaluator = Evaluator('vae', config, val_loader, model)

    optimizer = torch.optim.Adam(model.parameters(), config.hyperparameters['lr'])

    trainer.setCriterion(vae_mse_loss_function)
    trainer.setOptimizer(optimizer)
    if args.evaluate:
        evaluator.setCriterion(vae_mse_loss_function)

    # optionally resume from a checkpoint
    if args.resume:
        trainer.load_saved_checkpoint(checkpoint=None)

    # Turn on benchmark if the input sizes don't vary
    # It is used to find best way to run models on your machine
    cudnn.benchmark = True

    best_prec1 = 0
    for epoch in range(config.hyperparameters['num_epochs']):
        if args.distributed:
            train_sampler.set_epoch(epoch)

        trainer.adjust_learning_rate(epoch)
        trainer.train(epoch)
        trainer.step()

        if args.evaluate:
            # evaluate on validation set
            prec1 = evaluator.evaluate(epoch)

            # remember best prec@1 and save checkpoint
            is_best = prec1 > best_prec1
            best_prec1 = max(prec1, best_prec1)
            trainer.save_checkpoint({
                'epoch': epoch+1,
                'state_dict': model.state_dict(),
                'best_prec1': best_prec1,
                'optimizer': optimizer.state_dict(),
            }, is_best, checkpoint=None)
        else:
            trainer.save_checkpoint({
                'epoch': epoch+1,
                'state_dict': model.state_dict(),
                'best_prec1': best_prec1,
                'optimizer': optimizer.state_dict(),
            }, is_best, checkpoint=None)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Disentangling Variations', formatter_class=RawTextHelpFormatter)

    parser.add_argument('--gpu', type=int, default=0, \
                        help="Turn ON for GPU support; default=0")
    parser.add_argument('--resume', type=int, default=0, \
                        help="Turn ON to resume training from latest checkpoint; default=0")
    parser.add_argument('--chkpts', type=str, default="./checkpoints", \
                        help="Mention the dir that contains checkpoints")
    parser.add_argument('--config', type=str, required=True, \
                        help="Mention the file to load required configurations of the model")
    parser.add_argument('--seed', type=int, default=100, \
                        help="Seed for random function, default=100")
    parser.add_argument('--pretrained', type=int, default=0, \
                        help="Turn ON if checkpoints of model available in /checkpoints dir")
    parser.add_argument('--evaluate', type=int, default=1, \
                        help='evaluate model on validation set')
    args = parser.parse_args()

    main(args)
