import time
import os
import sys
import shutil
import wandb
from termcolor import cprint
from argparse import Namespace
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import cv2
import numpy as np

from JailNTL.data import create_dataset
from JailNTL.models import create_model
from JailNTL.utils.visualizer import Visualizer
from JailNTL.utils.util import setup_seed, tensor2im
from JailNTL.utils.visualizer import save_images
from JailNTL.data import load_disguised_data, jailntl_data_setup
from NTL.utils.utils import auto_save_name
from NTL.eval import eval_src
import NTL.utils.load_utils as load_utils
from NTL.utils import Cus_Dataset


def load_ntl_model(config):
    """Load NTL model with pretrained parameters."""
    # Init NTL model
    model_ntl = load_utils.load_model(config)
    
    # Load pretrained NTL parameters
    if config.NTL_save_path == 'auto':
        save_path = auto_save_name(config)
    else: 
        save_path = config.NTL_save_path
    model_ntl.load_state_dict(torch.load(save_path))
    
    # Freeze NTL parameters and set eval mode
    cprint(f'Load pretrainzed NTL model from {save_path}', 'green')
    for param in model_ntl.parameters():
        param.requires_grad = False
    model_ntl.eval()
    
    return model_ntl

def train_disguising_model(config, ntl, task_name, dataroot):  
    # Train
    dataset = create_dataset(config, 'jailntl_train', dataroot=dataroot)  # create a dataset given config.dataset_mode and other configions
    dataset_size = len(dataset)    # get the number of images in the dataset.
    print('The number of training images = %d' % dataset_size)
    model = create_model(config, mode='train', task_name=task_name, ntl_model=ntl) # create a model given config.model and other configions
    model.setup(config)               # regular setup: load and print networks; create schedulers
    visualizer = Visualizer(config, task_name)   # create a visualizer that display/save images and plots
    total_iters = 0                # the total number of training iterations
    
    # outer loop for different epochs; we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>
    for epoch in range(1, config.n_epochs + config.n_epochs_decay + 1):
        epoch_start_time = time.time()  # timer for entire epoch
        iter_data_time = time.time()    # timer for data loading per iteration
        epoch_iter = 0                  # the number of training iterations in current epoch, reset to 0 every epoch
        model.update_learning_rate()    # update learning rates in the beginning of every epoch.
        for i, data in enumerate(dataset):  # inner loop within one epoch
            iter_start_time = time.time()  # timer for computation per iteration
            if total_iters % config.print_iter_freq == 0:
                t_data = iter_start_time - iter_data_time

            total_iters += config.jailntl_batch_size
            epoch_iter += config.jailntl_batch_size
            model.set_input(data)         # unpack data from dataset and apply preprocessing
            model.optimize_parameters()   # calculate loss functions, get gradients, update network weights

            # print training losses and save logging information to the disk
            if total_iters % config.print_iter_freq == 0:
                losses = model.get_current_losses()
                t_comp = (time.time() - iter_start_time) / config.jailntl_batch_size
                visualizer.display_current_losses(epoch, epoch_iter, losses, t_comp, t_data)

            iter_data_time = time.time()
        if epoch % config.save_epoch_freq == 0:
            print('saving the model at the end of epoch %d, iters %d' % (epoch, total_iters))
            model.save_networks('latest')
            model.save_networks(epoch)

        print('End of epoch %d / %d \t Time Taken: %d sec' % (epoch, config.n_epochs + config.n_epochs_decay, time.time() - epoch_start_time))
    
    # copy G_A as G
    shutil.copy2(f'{config.checkpoints_dir}/{task_name}/latest_net_G_A.pth', f'{config.checkpoints_dir}/{task_name}/latest_net_G.pth')
   
def run_disguising_model(config, task_name, dataroot, results_dir):
    dataset = create_dataset(config, mode='jailntl_val', dataroot=dataroot)  # create a dataset given config.dataset_mode and other configions
    model = create_model(config, mode='test', task_name=task_name)      # create a model given config.model and other configions
    model.setup(config)               # regular setup: load and print networks; create schedulers
   
    # if config.eval:
    #     model.eval()
        
    disguised_imgs = []
    disguised_labels = []
    for i, data in enumerate(dataset):
        if i >= config.num_test:  # only apply our model to config.num_test images.
            break
        model.set_input(data)  # unpack data from data loader
        label = data['A_label']
        print(f'process image {i} with label {int(label)}')
        
        model.test()           # run inference
        visuals = model.get_current_visuals()  # get image results
        save_images(visuals, image_name=f"{i}_class_{int(label)}", image_path=results_dir)
        
        img = tensor2im(visuals['real'])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        disguised_imgs.append(img)
        disguised_labels.append(label)
    
    # Save disguised images and labels
    return [disguised_imgs, disguised_labels, len(disguised_imgs)]

def attack(config, disguised_tgt_dir, disguised_src_dir, mode='disguised'):
    disguised_src_data = load_disguised_data(config, disguised_src_dir, config.domain_src, mode)
    disguised_tgt_data = load_disguised_data(config, disguised_tgt_dir, config.domain_tgt, mode)

    disguised_src_datafile = Cus_Dataset(dataset_name=config.domain_src, mode='val',
                            dataset=disguised_src_data, begin_ind=0, size=1000,
                            spec_dataTransform=None,
                            config=config)
    disguised_tgt_datafile = Cus_Dataset(dataset_name=config.domain_tgt, mode='val',
                            dataset=disguised_tgt_data, begin_ind=0, size= 1000,
                            spec_dataTransform=None,
                            config=config)
    disguised_src_dataloader = DataLoader(disguised_src_datafile, batch_size=1,
                                shuffle=False, num_workers=config.num_workers,
                                drop_last=False)
    disguised_tgt_dataloader = DataLoader(disguised_tgt_datafile, batch_size=1,
                                shuffle=False, num_workers=config.num_workers,
                                drop_last=False)    
    disguised_dataloader_val = [disguised_src_dataloader, disguised_tgt_dataloader]
    print(f"Attack Performance ({mode}): ")
    att_src_acc, att_tgt_acc = eval_src(config, disguised_dataloader_val,
                        ntl_model, datasets_name=[config.domain_src, config.domain_tgt])
    attack_result = {
        'mode': mode,
        'src_acc': att_src_acc,
        'tgt_acc': att_tgt_acc
    }
    wandb.log(attack_result)
    

if __name__ == '__main__':
    # 1. Environment setup: load config, setup seed, and prepare data
    if len(sys.argv) < 5:
        print("Please input the domain pair! e.g. python src/jailntl.py -s cifar -t stl")
        sys.exit(0)
    domain_src, domain_tgt = sys.argv[2], sys.argv[4]
    if domain_src == 'visda_t' and domain_tgt == 'visda_v':
        wandb.init(project='JailNTL_CodeRelease', config='config/attack/visda.yml')
    elif domain_src == 'cifar' and domain_tgt == 'stl':
        wandb.init(project='JailNTL_CodeRelease', config='config/attack/cifarstl.yml')
    elif domain_src == 'stl' and domain_tgt == 'cifar':
        wandb.init(project='JailNTL_CodeRelease', config='config/attack/stlcifar.yml')
    config = wandb.config
    setup_seed(config.seed)
    
    # prepare data for JailNTL
    jailntl_data_setup(config)
    
    # setup dir and names
    tgt_dataroot = f"./data_presplit/jailntl/img/{config.domain_tgt}_2_{config.domain_src}_{config.jailntl_shot_num}-shot"
    src_dataroot = f"./data_presplit/jailntl/img/{config.domain_src}_2_{config.domain_src}_{config.jailntl_shot_num}-shot"
    tgt_name = f'{config.domain_tgt}_2_{config.domain_src}_{config.task_name}_{config.NTL_network}_{config.jailntl_shot_num}-shot_{config.confidence_weight}_{config.class_balance_weight}_{config.grad_epsilon}'
    src_name = f'{config.domain_src}_2_{config.domain_src}_{config.task_name}_{config.NTL_network}_{config.jailntl_shot_num}-shot_{config.confidence_weight}_{config.class_balance_weight}_{config.grad_epsilon}'
    tgt_results_dir = f'./disguised_results/tgt_results/{tgt_name}'
    src_results_dir = f'./disguised_results/src_results/{src_name}'
    
    # 2. load ntl
    ntl_model = None
    ntl_model = load_ntl_model(config)
    
    # 3. Train Disguising Model
    # tgt -> src
    train_disguising_model(config, ntl_model, task_name=tgt_name, dataroot=tgt_dataroot)
    # src -> src
    train_disguising_model(config, ntl_model, task_name=src_name, dataroot=src_dataroot)

    # 4. Run Disguising Model
    # tgt -> src
    if not os.path.exists(tgt_results_dir):
        os.makedirs(tgt_results_dir)
    disguised_data_tgt = run_disguising_model(config, task_name=tgt_name, dataroot=tgt_dataroot, results_dir=tgt_results_dir)
    # src -> src
    if not os.path.exists(src_results_dir):
        os.makedirs(src_results_dir)
    disguised_data_src = run_disguising_model(config, task_name=src_name, dataroot=src_dataroot, results_dir=src_results_dir)

    # 5. Attack NTL
    ntl_model = load_ntl_model(config)
    print(f"NTL Task: {config.domain_src} -> {config.domain_tgt}, {config.task_name}")
    attack(config, tgt_results_dir, src_results_dir, mode='disguised')
    attack(config, tgt_results_dir, src_results_dir, mode='real')