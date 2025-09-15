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

def get_visda_data_src():
    return get_visda_data(source='train')

def get_visda_data_tgt():
    return get_visda_data(source='validation')

def get_visda_data(source):
    list_img = []
    list_label = []
    data_size = 0
    root_temp = "./data/VisDA/{}".format(source)
    class_path = os.listdir(root_temp)
    for i in range(len(class_path)):
        class_temp = os.path.join(root_temp, class_path[i])
        img_path = os.listdir(class_temp)
        for j in range(1000):
            img_path_temp = os.path.join(class_temp, img_path[j])
            img = cv2.imread(img_path_temp)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (112, 112))
            
            list_img.append(img)
            list_label.append(np.eye(12)[i])
            data_size += 1

    # np.random.seed(0)
    ind = np.arange(data_size)
    ind = np.random.permutation(ind)
    
    list_img = np.asarray(list_img)
    list_img = list_img[ind]

    list_label = np.asarray(list_label)
    list_label = list_label[ind]
    
    return [list_img, list_label, data_size]

