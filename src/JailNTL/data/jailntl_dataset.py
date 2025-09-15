"""This module implements the classes for the JailNTL datasets, JailNTLTrainCusDataset and JailNTLGenCusDataset.
For data preprocessing, we apply:
    - resizing
    - random cropping (only in training)
    - random horizontal flipping (only in training)
    - normalization
"""
import random
import numpy as np
import torch.utils.data as data
from PIL import Image
import torchvision.transforms as transforms
import torch
import os

from JailNTL.data.image_folder import make_dataset


def get_transform(opt, phase='train'):
    """Get the image transformation for JailNTL dataset."""
    assert(phase in ['train', 'test'])
    transform_list = []
   
    # Resize
    if phase == 'train':
        osize = [opt.load_size, opt.load_size]
    elif phase == 'test':
        osize = [opt.crop_size, opt.crop_size]
    transform_list.append(transforms.Resize(osize, transforms.InterpolationMode.BICUBIC))
    
    # Data augmentation
    if phase == 'train':
        # Random crop
        transform_list.append(transforms.RandomCrop(opt.crop_size))
        # Random horizontal flipping
        transform_list.append(transforms.RandomHorizontalFlip())

    # Convert to tensor
    transform_list += [transforms.ToTensor()]
    
    # Normalize
    transform_list += [transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    
    return transforms.Compose(transform_list)

class JailNTLTrainCusDataset(data.Dataset):
    """
    The dataset class for the training phase of JailNTL.
    It assumes that the directory '/path/to/data/trainA' contains images from domain A (e.g., target domain),
    and '/path/to/data/trainB' contains images from domain B (e.g., source domain).
    The number of images in '/path/to/data/trainA' may be different from that in '/path/to/data/trainB'.
    """

    def __init__(self, opt, dir):
        """Initialize the class; save the options in the class

        Parameters:
            opt -- stores all the experiment flags;
            dir (str) -- path to the directory containing images
        """
        self.opt = opt
        
        self.dir_A = os.path.join(dir, 'trainA')
        self.dir_B = os.path.join(dir, 'trainB')
        self.A_paths = sorted(make_dataset(self.dir_A))   # load images from '/path/to/data/trainA'
        self.B_paths = sorted(make_dataset(self.dir_B))    # load images from '/path/to/data/trainB'
        self.A_size = len(self.A_paths)
        self.B_size = len(self.B_paths)
        
        self.transform_A = get_transform(self.opt, phase='train')
        self.transform_B = get_transform(self.opt, phase='train')

    def __len__(self):
        """Return the total number of images in the dataset."""
        return min(self.A_size, self.B_size)

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index - - a random integer for data indexing

        Returns:
            a dictionary of data with their names. It contains the data itself and its metadata information.
        """
        A_path = self.A_paths[index % self.A_size]  # make sure index is within then range
        index_B = random.randint(0, self.B_size - 1)  # randomize the index for domain B to avoid fixed pairs
        B_path = self.B_paths[index_B]
        A_img = Image.open(A_path).convert('RGB')
        B_img = Image.open(B_path).convert('RGB')
        # apply image transformation
        A = self.transform_A(A_img)
        B = self.transform_B(B_img)

        return {'A': A, 'B': B, 'A_paths': A_path, 'B_paths': B_path}

class JailNTLGenCusDataset(data.Dataset):
    """This dataset class can be used for the validation phase of JailNTL.
    It assumes that the directory '/path/to/data/testA' contains images from domain A (e.g., target domain).
    """

    def __init__(self, opt, dir):
        """Initialize the class; save the options in the class

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        self.opt = opt
        self.dir_A = os.path.join(dir, 'testA')  # create a path '/path/to/data/testA'
        self.A_paths = sorted(make_dataset(self.dir_A))
        self.transform = get_transform(self.opt, phase='test')

    def __len__(self):
        """Return the total number of images in the dataset."""
        return len(self.A_paths)

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index - - a random integer for data indexing

        Returns:
            a dictionary of data with their names. It ususally contains the data itself and its metadata information.
        """
        A_path = self.A_paths[index]
        A_img = Image.open(A_path).convert('RGB')
        A = self.transform(A_img)
        A_label = int(A_path.split('/')[-1].split('_')[2].split('.')[0])  # xx/no_classe_{label}.jpg
        
        return {'A': A, 'A_paths': A_path, 'A_label': A_label}

