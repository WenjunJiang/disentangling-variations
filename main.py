#!/usr/bin/env python

import os
import sys
import argparse
from argparse import RawTextHelpFormatter
import time
import shutil

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
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def count_parameters(model):
    """Counts trainable(active) parameters of a model"""
    total_params = sum(reduce( lambda a, b: a*b, x.size()) for x in model.parameters())
    return total_params

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def adjust_learning_rate(optimizer, epoch, config):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = config.hyperparameters['lr'] * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

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

def loss_function(recon_x, x, mu, logvar):
    bce = F.binary_cross_entropy(recon_x, x.view(-1, 784), size_average=False)
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return bce + kld
    
def vae_loss_function(recon_x, target, mu, logvar):
    # BCE = F.binary_cross_entropy(recon_x, x.view(-1, 784), size_average=False)
    mse_loss = nn.MSELoss(size_average=False)
    MSE = mse_loss(recon_x.view(-1, 196608), target.view(-1, 196608))
    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return MSE + KLD

class Trainer(object):
    """docstring for Trainer."""
    def __init__(self, config, data, models):
        super(Trainer, self).__init__()
        self.config = config
        self.data = data
        self.epoch = 0

        # models
        self.vae = models[0]
        self.vae_train_loss = 0
        self.lat_dis = models[1]
        self.ptc_dis = models[2]
        # self.clf_dis = models[3]

        # criterion & optimizers
        self.vae_criterion = nn.CrossEntropyLoss().to(device)
        self.vae_optimizer = torch.optim.Adam(self.vae.parameters(), self.config.hyperparameters['lr'],
                                        weight_decay=self.config.hyperparameters['weight_decay'])

        self.lat_dis_criterion = nn.CrossEntropyLoss().to(device)
        self.lat_dis_optimizer = torch.optim.Adam(self.lat_dis.parameters(), self.config.hyperparameters['lr'],
                                        weight_decay=self.config.hyperparameters['weight_decay'])

        self.ptc_dis_criterion = nn.CrossEntropyLoss().to(device)
        self.ptc_dis_optimizer = torch.optim.Adam(self.ptc_dis.parameters(), self.config.hyperparameters['lr'],
                                        weight_decay=self.config.hyperparameters['weight_decay'])

        # reload pretrained models

        # training stats
        self.stats = {}
        self.stats['vae_training_logs'] = []
        self.stats['lat_dis_training_logs'] = []
        self.stats['ptc_dis_training_logs'] = []

        # best reconstruction loss / best accuracy
        self.best_loss = 1e12
        self.best_accu = -1e12
        self.n_total_iter = 0

    def lat_dis_step(self):
        pass

    def ptc_dis_step(self):
        pass

    def clf_dis_step(self):
        pass

    def vae_step(self, epoch):
        # switch to train mode
        self.vae.train()
        self.train_loss = 0
        for batch_idx, (images, labels) in enumerate(self.data):
            if self.config.gpu:
                images = images.to(device)
                labels = labels.to(device)
            recon_x, mu, logvar = self.vae(images)
            loss = vae_loss_function(recon_x, images, mu, logvar)
            self.vae_optimizer.zero_grad()
            loss.backward()
            self.train_loss += loss.item()
            self.vae_optimizer.step()

            if batch_idx % self.config.logs['log_interval'] == 0:
                print(
                    'Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch+1, batch_idx * len(images), len(self.data.dataset),
                    100. * batch_idx / len(self.data),
                    loss.item() / len(self.data))
                )

    def step(self, epoch):
        self.epoch = epoch


    def save_checkpoint(self, epoch, best_prec, is_best):
        state = {
            'epoch': epoch+1,
            'state_dict': self.vae.state_dict(),
            'best_prec1': best_prec,
            'optimizer': self.vae_optimizer.state_dict()
        }
        ckpt_path = os.path.join(self.config.checkpoints['loc'], self.config.checkpoints['ckpt_fname'])
        best_ckpt_path = os.path.join(self.config.checkpoints['loc'], \
                            self.config.checkpoints['best_ckpt_fname'])
        torch.save(state, ckpt_path)
        if is_best:
            shutil.copy(ckpt_path, best_ckpt_path)

    def adjust_learning_rate(self):
        lr = self.config.hyperparameters['lr'] * (0.1 ** (self.epoch // 30))
        for param_group in self.optim.param_groups:
            param_group['lr'] = lr

    def load_model(self, model, optim):
        checkpoint = torch.load(os.path.join(self.config.checkpoints['loc'], \
                        self.config.checkpoints['ckpt_fname']))

        start_epoch = checkpoint['epoch']
        best_prec1 = checkpoint['best_prec1']
        model.load_state_dict(checkpoint['state_dict'])
        optim.load_state_dict(checkpoint['optimizer'])

        print("[#] Loaded Checkpoint '{}' (epoch {})"
            .format(self.config.checkpoints['ckpt_fname'], checkpoint['epoch']))
        return (start_epoch, best_prec1)

class Evaluator(object):
    """docstring for Evaluate."""
    def __init__(self, config, data, models):
        super(Evaluator, self).__init__()
        self.config = config
        self.data = data
        self.eval_loss = 0

        # models
        self.vae = models[0]
        self.lat_dis = models[1]
        self.ptc_dis = models[2]
        self.clf_dis = models[3]
        self.eval_clf = models[4]

        assert self.eval_clf.image_size == config.data['image_size']
        assert all(attr in self.eval_clf.attributes for attr in config.data['attributes'])

    def eval_vae_loss(self, epoch):
        # switch to eval mode
        self.vae.eval()
        self.eval_loss = 0
        with torch.no_grad():
            for batch_idx, (images, labels) in enumerate(self.data):
                if self.config.gpu:
                    images = images.to(device)
                    labels = labels.to(device)

                # compute output
                recon_x, mu, logvar = self.vae(images)
                self.eval_loss += vae_loss_function(recon_x, images, mu, logvar).item()

                # save decoder output to check vae's generated images
                if batch_idx == 0:
                    n = min(images.size(0), 8)
                    img_sz = self.config.data['image_size']
                    comparison = torch.cat([images[:n], \
                        recon_x.view(self.config.data['batch_size'], 3, img_sz, img_sz)[:n]])
                    img_nm = os.path.join(self.config.data['generated'], \
                        'reconstruction_' + str(epoch) + '.png')
                    save_image(comparison.cpu(), img_nm, nrow=n)

        self.eval_loss /= len(self.data.dataset)
        return self.eval_loss

    def eval_reconstruction_loss(self):
        pass

    def eval_lat_dis_accuracy(self):
        pass

    def eval_ptc_dis_accuracy(self):
        pass

    def eval_clf_dis_accuracy(self):
        pass

    def eval_clf_accuracy(self):
        pass

    def evaluate(self, n_epoch):
        pass

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

    vae = VAE(config.hyperparameters)
    lat_dis = LatentDiscriminator(config.hyperparameters)
    ptc_dis = PatchDiscriminator(config.hyperparameters)
    if not args.distributed:
        vae = nn.DataParallel(vae).to(device)
        lat_dis = nn.DataParallel(lat_dis).to(device)
        ptc_dis = nn.DataParallel(ptc_dis).to(device)
    else:
        vae.to(device)
        vae = nn.parallel.DistributedDataParallel(vae)
        lat_dis.to(device)
        lat_dis = nn.parallel.DistributedDataParallel(lat_dis)
        ptc_dis.to(device)
        ptc_dis = nn.parallel.DistributedDataParallel(ptc_dis)

    cudnn.benchmark = True

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

    trainer = Trainer(config, train_loader, \
                (vae, lat_dis, ptc_dis))
    if args.evaluate:
        evaluator = Evaluator(config, val_loader, \
                    (vae, lat_dis, ptc_dis))
        evaluator = Evaluator(config, val_loader, vae)

    for epoch in range(config.hyperparameters['num_epochs']):
        time1 = time.time()
        if args.distributed:
            train_sampler.set_epoch(epoch)

        # latent discriminator training
        for _ in range(config.hyperparameters['lat_dis_steps']):
            trainer.lat_dis_step()

        # patch discriminator training
        for _ in range(config.hyperparameters['ptc_dis_steps']):
            trainer.ptc_dis_step()

        # classifier discriminator training
        for _ in range(config.hyperparameters['clf_dis_steps']):
            trainer.clf_dis_step()

        trainer.vae_step(epoch)
        trainer.step(epoch)

        if args.evaluate:
            # evaluate on validation set
            prec1 = evaluator.eval_vae_loss(epoch)
            # remember best prec@1 and save checkpoint
            is_best = prec1 > best_prec1
            best_prec1 = max(prec1, best_prec1)
            trainer.save_checkpoint(epoch, best_prec1, is_best)
        else:
            trainer.save_checkpoint(epoch, 0, True)
            
        time2 = time.time()
        print('[+] Time taken by this epoch is {}'.format(time2-time1))

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
