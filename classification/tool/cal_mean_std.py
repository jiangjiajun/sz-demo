# -*- coding: UTF-8 -*-
import argparse
import functools
import os
import cv2
import numpy as np
import multiprocessing

class CalMeanStd(object):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        
    def cal_per_img(self, line):
        part = line.split(' ')
        img = cv2.imread(os.path.join(self.data_dir, part[0]), 1)
        print('The image path: ' + os.path.join(self.data_dir, part[0]))
        return np.mean(img[:,:,0]), np.mean(img[:,:,1]), np.mean(img[:,:,2])

    def calculate(self, flist):
        items = [line.strip() for line in flist]
        p = multiprocessing.Pool(6)
        out = p.map(self.cal_per_img, items)
        p.close()
        p.join()
        out = np.array(out)
        per_image_Bmean = out[:, 0]
        per_image_Gmean = out[:, 1]
        per_image_Rmean = out[:, 2]
        R_mean = np.mean(per_image_Rmean)
        G_mean = np.mean(per_image_Gmean)
        B_mean = np.mean(per_image_Bmean)
        R_std = np.std(per_image_Rmean)
        G_std = np.std(per_image_Gmean)
        B_std = np.std(per_image_Bmean)
        return [R_mean/255.0, G_mean/255.0, B_mean/255.0], \
               [R_std/255.0, G_std/255.0, B_std/255.0]
    

def add_arguments(argname, type, default, help, argparser, **kwargs):
    """Add argparse's argument.

    Usage:

    .. code-block:: python

        parser = argparse.ArgumentParser()
        add_argument("name", str, "Jonh", "User name.", parser)
        args = parser.parse_args()
    """
    type = distutils.util.strtobool if type == bool else type
    argparser.add_argument(
        "--" + argname,
        default=default,
        type=type,
        help=help + ' Default: %(default)s.',
        **kwargs)

parser = argparse.ArgumentParser(description=__doc__)
add_arg = functools.partial(add_arguments, argparser=parser)
add_arg('data_dir',       str,     "./data/ILSVRC2012/",    "The dataset root directory.")

def mean_std(args):
    mean_std_path = os.path.join(args.data_dir, "mean_std.txt")
    train_txt = os.path.join(args.data_dir, "train_list.txt")
    per_image_Rmean = []
    per_image_Gmean = []
    per_image_Bmean = []
    with open(train_txt, "r") as flist:
        cal_meanstd = CalMeanStd(args.data_dir)
        mean, std = cal_meanstd.calculate(flist)
    with open(mean_std_path, "w") as fw:
        fw.write(str(mean) + "\n")
        fw.write(str(std) + "\n")

def main():
    args = parser.parse_args()
    mean_std(args)

if __name__ == '__main__':
    main()