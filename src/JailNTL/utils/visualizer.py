import numpy as np
import os
import sys
import ntpath
import time
from . import util


def save_images(visuals, image_name, image_path):
    """Save images to the disk.

    Parameters:
        visuals (OrderedDict)    -- an ordered dictionary that stores (name, images (either tensor or numpy) ) pairs
        image_path (str)         -- the string is used to create image paths
        image_name (str)         -- the string will be used to save the image with the name <image_path>/<image_name>_label.png
    
    This function will save images to 'image_path/image_name_label.png'
    """

    for label, im_data in visuals.items():
        im = util.tensor2im(im_data)
        image_name_full = '%s_%s.png' % (image_name, label)
        save_path = os.path.join(image_path, image_name_full)
        util.save_image(im, save_path)


class Visualizer():
    """This class includes several functions that can display/save images and print/save logging information.

    It uses a Python library 'visdom' for display, and a Python library 'dominate' (wrapped in 'HTML') for creating HTML files with images.
    """

    def __init__(self, opt, task_name):
        """Initialize the Visualizer class

        Parameters:
            opt -- stores all the experiment flags; needs to be a subclass of BaseOptions
        Step 1: Cache the training/test options
        Step 2: create a logging file to store training losses
        """
        self.opt = opt 
        self.log_name = os.path.join(opt.checkpoints_dir, task_name, 'loss_log.txt')
        if not os.path.exists(os.path.dirname(self.log_name)):
            os.makedirs(os.path.dirname(self.log_name))
        with open(self.log_name, "a") as log_file:
            now = time.strftime("%c")
            log_file.write('================ Training Loss (%s) ================\n' % now)

    # losses: same format as |losses| of plot_current_losses
    def display_current_losses(self, epoch, iters, losses, t_comp, t_data):
        """print current losses on console; also save the losses to the disk

        Parameters:
            epoch (int) -- current epoch
            iters (int) -- current training iteration during this epoch (reset to 0 at the end of every epoch)
            losses (OrderedDict) -- training losses stored in the format of (name, float) pairs
            t_comp (float) -- computational time per data point (normalized by batch_size)
            t_data (float) -- data loading time per data point (normalized by batch_size)
        """
        message = '(epoch: %d, iters: %d, time: %.3f, data: %.3f) ' % (epoch, iters, t_comp, t_data)
        for k, v in losses.items():
            message += '%s: %.3f ' % (k, v)

        print(message)  # print the message
        with open(self.log_name, "a") as log_file:
            log_file.write('%s\n' % message)  # save the message
