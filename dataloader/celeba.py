#!/usr/bin/env python

import os
import sys
from os import listdir
from os.path import isfile, join
import re
import numpy as np

from PIL import Image
import torchvision
from torchvision import transforms
import torch
from torch.utils.data.dataset import Dataset
from torch.utils.data.sampler import SubsetRandomSampler
import cv2

AVAILABLE_ATTR = [
    "5_o_Clock_Shadow", "Arched_Eyebrows", "Attractive", "Bags_Under_Eyes", "Bald",
    "Bangs", "Big_Lips", "Big_Nose", "Black_Hair", "Blond_Hair", "Blurry", "Brown_Hair",
    "Bushy_Eyebrows", "Chubby", "Double_Chin", "Eyeglasses", "Goatee", "Gray_Hair",
    "Heavy_Makeup", "High_Cheekbones", "Male", "Mouth_Slightly_Open", "Mustache",
    "Narrow_Eyes", "No_Beard", "Oval_Face", "Pale_Skin", "Pointy_Nose",
    "Receding_Hairline", "Rosy_Cheeks", "Sideburns", "Smiling", "Straight_Hair",
    "Wavy_Hair", "Wearing_Earrings", "Wearing_Hat", "Wearing_Lipstick",
    "Wearing_Necklace", "Wearing_Necktie", "Young"
]

# TODO: Create logger & fix this
def log_attributes_stats(train_attributes, valid_attributes, test_attributes, config):
    """
    Log attributes distributions.
    """
    k = 0
    for n_cat, attr_name in enumerate(config['attributes']):
        # logger.debug('Train %s: %s' % (attr_name, ' / '.join(['%.5f' % train_attributes[:, k + i].mean() for i in range(n_cat)])))
        # logger.debug('Valid %s: %s' % (attr_name, ' / '.join(['%.5f' % valid_attributes[:, k + i].mean() for i in range(n_cat)])))
        # logger.debug('Test  %s: %s' % (attr_name, ' / '.join(['%.5f' % test_attributes[:, k + i].mean() for i in range(n_cat)])))
        # assert train_attributes[:, k:k + n_cat].sum() == train_attributes.size(0)
        # assert valid_attributes[:, k:k + n_cat].sum() == valid_attributes.size(0)
        # assert test_attributes[:, k:k + n_cat].sum() == test_attributes.size(0)
        # k += n_cat
        print((n_cat, attr_name))
    # assert k == len(config.attributes)

def load_celeba_images(config):
    loc = config['root']
    if config['type'] == 'reg':
        loc = os.path.join(loc, 'celeba_processed')
    elif config['type'] == 'align':
        loc = os.path.join(loc, 'celeba_align_processed')
    elif config['type'] == 'hq':
        # TODO: hq not available yet
        loc = os.path.join(loc, 'celeba_align_processed')
    else:
        print('[-] Incorrect input')
        return
    all_images = [f for f in listdir(loc) if isfile(join(loc, f))]

    attrs = []
    for n_cat, name in enumerate(config['attributes']):
        for i in range(n_cat):
            attrs.append(torch.FloatTensor((attributes[name] == i).astype(np.float32)))
    attributes = torch.cat([x.unsqueeze(1) for x in attrs], 1)

    # splitting train, valid, and test data
    train_index = 162770
    valid_index = train_index + 19867
    test_index =  len(all_images)

    train_images = all_images[:train_index]
    valid_images = all_images[train_index:valid_index]
    test_images = all_images[valid_index:test_index]
    train_attributes = attributes[:train_index]
    valid_attributes = attributes[train_index:valid_index]
    test_attributes = attributes[valid_index:test_index]

    log_attributes_stats(train_attributes, valid_attributes, test_attributes, config)
    images = (train_images, valid_images, test_images)
    attributes = (train_attributes, valid_attributes, test_attributes)

    return images, attributes

def normalize_images(images):
    """Normalize image values."""
    return images.float().div_(255.0).mul_(2.0).add_(-1)

class CelebaDataset(Dataset):
    def __init__(self, images, attributes, config, transform=None):
        """
        Initialize the data sampler with training data.
        """
        assert len(images) == len(attributes)
        self.images = images
        self.attributes = attributes
        self.batch_size = config.data['batch_size']
        self.v_flip = config.data['v_flip']
        self.h_flip = config.data['h_flip']
        self.transform = transform
        self.config = config        

    def __len__(self):
        """
        Number of images in the object dataset.
        """
        return len(self.images)

    def __getitem__(self, index):
        root = self.config.data['root']
        if self.config.data['type'] == 'reg':
            root = os.path.join(root, 'celeba_processed')
        elif self.config.data['type'] == 'align':
            root = os.path.join(root, 'celeba_align_processed')
        elif self.config.data['type'] == 'hq':
            root = os.path.join(root, 'celeba_align_processed')

        image_path = os.path.join(root, self.images[index])
        image = Image.open(image_path).convert('RGB')

        if self.transform is not None:
            image = self.transform(image)
        
        attribute = self.attributes[index]

        return (attribute, image)