import numpy as np
from collections.abc import Iterable
import os
from PIL import Image
import matplotlib.pyplot as plt
import pickle
import math
from scipy.special import softmax
import scipy.io as sio
import torchvision.datasets as datasets
import cv2


def rgb_loader(path):
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')

def get_cifar_data():
    list_img = []
    list_label = []
    data_size = 0
    dir = './data/cifar-10-batches-py'

    for filename in ['%s/data_batch_%d' % (dir,j) for j in range(1, 6)]:
        with open(filename, 'rb') as fo:
            cifar10 = pickle.load(fo, encoding = 'bytes')
        for i in range(len(cifar10[b"labels"])):
            img = np.reshape(cifar10[b"data"][i], (3,32,32))
            img = np.transpose(img, (1, 2, 0))
            #img = img.astype(float)
            list_img.append(img)
            
            list_label.append(np.eye(10)[cifar10[b"labels"][data_size%10000]])
            data_size += 1

    ind = np.arange(data_size)
    ind = np.random.permutation(ind)
    print('cifar random idx:')
    print(ind)
    list_img = np.asarray(list_img)
    list_img = list_img[ind]

    list_label = np.asarray(list_label)
    list_label = list_label[ind]

    return [list_img, list_label, data_size]

def get_stl_data():
    list_img = []
    list_label = []
    data_size = 0
    re_label = [0, 2, 1, 3, 4, 5, 7, 6, 8, 9]
    root = './data/stl10_binary'
    train_x_path = os.path.join(root, 'train_X.bin')
    train_y_path = os.path.join(root, 'train_y.bin')
    test_x_path = os.path.join(root, 'test_X.bin')
    test_y_path = os.path.join(root, 'test_y.bin')


    with open(train_x_path, 'rb') as fo:
        train_x = np.fromfile(fo, dtype=np.uint8)
        train_x = np.reshape(train_x, (-1, 3, 96, 96))
        train_x = np.transpose(train_x, (0, 3, 2, 1))
    with open(train_y_path, 'rb') as fo:
        train_y = np.fromfile(fo, dtype=np.uint8)

    for i in range(len(train_y)):
        label = re_label[train_y[i] - 1]
        list_img.append(train_x[i])
        list_label.append(np.eye(10)[label])
        data_size += 1

    with open(test_x_path, 'rb') as fo:
        test_x = np.fromfile(fo, dtype=np.uint8)
        test_x = np.reshape(test_x, (-1, 3, 96, 96))
        test_x = np.transpose(test_x, (0, 3, 2, 1))
    with open(test_y_path, 'rb') as fo:
        test_y = np.fromfile(fo, dtype=np.uint8)

    for i in range(len(test_y)):
        label = re_label[test_y[i] - 1]
        list_img.append(test_x[i])
        list_label.append(np.eye(10)[label])
        data_size += 1
    
    ind = np.arange(data_size)
    ind = np.random.permutation(ind)
    print('stl random idx:')
    print(ind)
    list_img = np.asarray(list_img)
    list_img = list_img[ind]

    list_label = np.asarray(list_label)
    list_label = list_label[ind]

    return [list_img, list_label, data_size]

