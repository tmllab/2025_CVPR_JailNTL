from collections import defaultdict
from torchvision.datasets import MNIST
import xml.etree.ElementTree as ET
from zipfile import ZipFile
import argparse
import tarfile
import gdown
import os


# utils
def stage_path(data_dir, name):
    full_path = os.path.join(data_dir, name)

    if not os.path.exists(full_path):
        os.makedirs(full_path)

    return full_path

def download_and_extract(url, dst, remove=True):
    gdown.download(url, dst, quiet=False)

    if dst.endswith((".tar.gz", ".tgz")):
        tar = tarfile.open(dst, "r:gz")
        tar.extractall(os.path.dirname(dst))
        tar.close()

    if dst.endswith(".tar"):
        tar = tarfile.open(dst, "r:")
        tar.extractall(os.path.dirname(dst))
        tar.close()

    if dst.endswith(".zip"):
        zf = ZipFile(dst, "r")
        zf.extractall(os.path.dirname(dst))
        zf.close()

    if remove:
        os.remove(dst)

# CIFAR10 
def download_cifar(data_dir, remove=True):
    download_and_extract("https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz",
                         os.path.join(data_dir, "cifar-10-python.tar.gz"))
    
# STL10
def download_stl(data_dir, remove=True):
    download_and_extract("http://ai.stanford.edu/~acoates/stl10/stl10_binary.tar.gz",
                         os.path.join(data_dir, "stl10_binary.tar.gz"))
    
# VisDA Train and Validation
def download_visda(data_dir, remove=True):
    full_path = stage_path(data_dir, "VisDA")

    download_and_extract("http://csr.bu.edu/ftp/visda17/clf/train.tar",
                         os.path.join(data_dir, "VisDA", "train.tar"))
    os.remove(os.path.join(full_path, 'train', 'image_list.txt'))
    download_and_extract("http://csr.bu.edu/ftp/visda17/clf/validation.tar",
                         os.path.join(data_dir, "VisDA", "validation.tar"))
    os.remove(os.path.join(full_path, 'validation', 'image_list.txt'))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Download datasets')
    parser.add_argument('--data_dir', type=str, default='./data')
    args = parser.parse_args()

    download_cifar(args.data_dir)
    download_stl(args.data_dir)
    download_visda(args.data_dir)

