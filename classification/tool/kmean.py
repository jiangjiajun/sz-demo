# -*- coding: UTF-8 -*-
import cv2
import os
import numpy as np
import sys
import pickle
import argparse
import functools
import random as rdm
#import matplotlib
#matplotlib.use('TkAgg')
from PCV.tools import imtools
from scipy import *
from pylab import *
from scipy.cluster.vq import *
from PCV.tools import pca


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

add_arg('num_class',       int,  'None',                 "number of class")
add_arg('data_dir',       str,  'None',                 "Directory of image")
add_arg('output_dir',       str,  'None',                 "Directory of output")

def start_kmeans(args):
    imlist = imtools.get_imlist(args.data_dir)

    imnbr = len(imlist)  # get the number of images
    print("The number of images is %d" % imnbr)

    # Create matrix to store all flattened images
    immatrix = np.array([np.array(cv2.imread(imname)).flatten() for imname in imlist], 'f')

    # PCA reduce dimension
    V, S, immean = pca.pca(immatrix)

    immatrix = np.array([np.array(cv2.imread(im)).flatten() for im in imlist])
    immatrix_src = np.array([np.array(cv2.imread(im)) for im in imlist])

    # project on the 40 first PCs
    immean = immean.flatten()
    projected = np.array([dot(V[:40], immatrix[i] - immean) for i in range(imnbr)])

    # k-means
    projected = whiten(projected)
    centroids, distortion = kmeans(projected, args.num_class)
    code, distance = vq(projected, centroids)
   
    output_path = os.path.join(args.output_dir, 'kmeans_result') 
    if not os.path.exists(output_path):
        os.mkdir(output_path)
    # plot clusters
    for k in range(args.num_class):
        ind = where(code == k)[0]
        print("class:", k, len(ind))
        os.mkdir(os.path.join(output_path, str(k)))
        i = rdm.randint(0,len(ind))
        cv2.imwrite(os.path.join(os.path.join(output_path, str(k)), str(i) + '.jpg'), immatrix_src[ind[i]])

def main():
    args = parser.parse_args()
    start_kmeans(args)


if __name__ == '__main__':
    main()


