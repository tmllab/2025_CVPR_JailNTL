"""This package includes all the modules related to data loading and preprocessing.
"""
import numpy as np
import cv2
import os
import torch.utils.data
from .jailntl_dataset import JailNTLTrainCusDataset, JailNTLGenCusDataset


def jailntl_data_split(datafile, num, save_path=None):
    """Split the data for jailbreaking NTL with a certain number of shots."""
    list_img, list_label = datafile.list_img, datafile.list_label
    
    # Shuffle the images and labels
    np.random.seed(2024)
    ind = np.arange(len(list_img))
    ind = np.random.permutation(ind)
    list_img = np.asarray(list_img)
    list_img = list_img[ind]
    list_label = list_label[ind]
    
    # Take a shot number of samples
    if num != -1:  # if num is -1, take all samples
        list_img = list_img[:num]
        list_label = list_label[:num]
    else:
        num = len(list_img)
    
    save_dir = f'data_presplit/jailntl/img/{save_path}'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    for i in range(num):
        img = list_img[i]
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        label = list_label[i]
        label = int(np.argmax(label))
        file_name = f'{i}_class_{label}.png'
        file_path = os.path.join(save_dir, file_name)
        cv2.imwrite(file_path, img)
        
    return
    
def jailntl_data_setup(config):
    """
    Prepare the data for jailbreaking NTL. Including:
        1) tgt -> src disguise, prepare the data for training and testing
        2) src -> src disguise, prepare the data for training and testing
    The data is saved in 'data_presplit/jailntl/img/', where:
        1) trainA, including few-shot target domain testing images (for tgt -> src disguise) or few-shot source domain testing images (for src -> src disguise)
        2) trainB, including few-shot source domain training images (for tgt -> src disguise) or few-shot source domain training images (for src -> src disguise)
        3) testA, including all target domain testing images (for tgt -> src disguise) or all source domain testing images (for src -> src disguise)
    """
    src_filename = f'data_presplit/{config.domain_src}_{config.image_size}.pth'
    tgt_filename = f'data_presplit/{config.domain_tgt}_{config.image_size}.pth'
    src_data = torch.load(src_filename, weights_only=False)
    tgt_data = torch.load(tgt_filename, weights_only=False)
    save_dir = 'data_presplit/jailntl'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    # Split the data for jailbreaking NTL
    # tgt -> src disguise
    src_datafile_train = src_data['train']
    tgt_datafile_val = tgt_data['val']
    tgt_data_dir = f'{config.domain_tgt}_2_{config.domain_src}_{config.jailntl_shot_num}-shot'
    jailntl_data_split(tgt_datafile_val, -1, f'{tgt_data_dir}/testA')
    jailntl_data_split(tgt_datafile_val, config.jailntl_shot_num, f'{tgt_data_dir}/trainA')
    jailntl_data_split(src_datafile_train, config.jailntl_shot_num, f'{tgt_data_dir}/trainB')

    # src -> src disguise
    src_datafile_val = src_data['val']
    src_data_dir = f'{config.domain_src}_2_{config.domain_src}_{config.jailntl_shot_num}-shot'
    jailntl_data_split(src_datafile_val, -1, f'{src_data_dir}/testA')
    jailntl_data_split(src_datafile_val, config.jailntl_shot_num, f'{src_data_dir}/trainA')
    jailntl_data_split(src_datafile_train, config.jailntl_shot_num, f'{src_data_dir}/trainB')

    print(f"Data split for jailbreaking NTL is saved in {save_dir}")

def load_disguised_data(config, file_dir, dataset_name, mode):
    """
    Load the disguised data for NTL test.
    All data preprocessing methods keep consistent with the NTL training phase.
    """
    image_size_dict = {'cifar': 32, 'stl': 96, 'visda_t': 112, 'visda_v': 112}
    file_suffix = 'fake.png' if mode == 'disguised' else 'real.png'
    
    class_num = config.num_classes
    image_size = image_size_dict[dataset_name]
    
    disguised_img = []
    disguised_labels = []
    disguised_size = 0
    for file_name in os.listdir(file_dir):
        if file_name.endswith(file_suffix):
            file_path = os.path.join(file_dir, file_name)

            label = int(file_name.split('_')[2])
            label = np.eye(class_num)[label]
            
            img = cv2.imread(file_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (image_size, image_size))
            disguised_img.append(img)
            disguised_labels.append(label)
            disguised_size += 1
    
    return [disguised_img, disguised_labels, disguised_size]

def create_dataset(opt, mode, dataroot):
    """Create a dataset given the options."""
    data_loader = JailNTLCusDatasetDataLoader(opt, mode, dataroot)
    dataset = data_loader.load_data()
    return dataset


class JailNTLCusDatasetDataLoader():
    """Wrapper class of Dataset class that performs multi-threaded data loading"""

    def __init__(self, opt, mode, dataroot):
        """Initialize this class

        Step 1: create a dataset instance given the name [dataset_mode]
        Step 2: create a multi-threaded data loader.
        """
        self.opt = opt
        if mode == 'jailntl_train':
            self.dataset = JailNTLTrainCusDataset(opt, dataroot)
            self.dataloader = torch.utils.data.DataLoader(
                self.dataset,
                batch_size=opt.jailntl_batch_size,
                shuffle=True,
                num_workers=int(opt.jailntl_num_threads))
        elif mode == 'jailntl_val':
            self.dataset = JailNTLGenCusDataset(opt, dataroot)
            self.dataloader = torch.utils.data.DataLoader(
                self.dataset,
                batch_size=1,
                shuffle=False,
                num_workers=0)
        print("dataset [%s] was created" % type(self.dataset).__name__)

    def load_data(self):
        return self

    def __len__(self):
        """Return the number of data in the dataset"""
        return len(self.dataset)

    def __iter__(self):
        """Return a batch of data"""
        for _, data in enumerate(self.dataloader):
            yield data

