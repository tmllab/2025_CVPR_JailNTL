import torch.utils.data as data
import torch
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
from torch.utils.data import DataLoader


def rgb_loader(path):
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')


class Cus_Dataset(data.Dataset):
    def __init__(self, dataset_name=None, mode = None, \
                            dataset = None, begin_ind = 0, size = 0, jailntl_shot_num = 0,
                            spec_dataTransform = None, config=None):

        self.mode = mode
        self.list_img = []
        self.list_label = []
        self.data_size = 0
        if spec_dataTransform is None: 
            dataTransform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((config.image_size, config.image_size)),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
            ])
            self.transform = dataTransform
        else: 
            self.transform = spec_dataTransform

        if self.mode == 'train':  # for ntl train, img-label pairs with shuffle
            self.data_size = size
            self.list_img = dataset[0][begin_ind: begin_ind+size]
            self.list_label = dataset[1][begin_ind: begin_ind+size]
            
            ind = np.arange(self.data_size)
            ind = np.random.permutation(ind)
            self.list_img = self.list_img[ind]
            self.list_label = np.asarray(self.list_label)
            self.list_label = self.list_label[ind]
        
        elif self.mode == 'val':  # for ntl val, img-label pairs without shuffle
            self.data_size = size
            self.list_img = dataset[0][begin_ind: begin_ind+size]
            self.list_label = dataset[1][begin_ind: begin_ind+size]
    

    def __getitem__(self, item):
        if self.mode == 'train':
            img = self.list_img[item]
            label = self.list_label[item]
            return self.transform(img), torch.LongTensor(label)
        elif self.mode == 'val':
            img = self.list_img[item]
            # img = np.array(self.list_img[item])
            label = self.list_label[item]
            # label = np.array(self.list_label[item])
            return self.transform(img), torch.LongTensor(label).unsqueeze(0)


    def __len__(self):
        return self.data_size

