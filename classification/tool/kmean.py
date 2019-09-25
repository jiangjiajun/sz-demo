# -*- coding: UTF-8 -*-
import cv2
import os
from PCV.tools import imtools
import pickle
from scipy import *
from pylab import *
from scipy.cluster.vq import *
from PCV.tools import pca
import numpy as np
import sys

_file = sys.argv[1]
imlist = imtools.get_imlist(str(_file))

imnbr = len(imlist)  # get the number of images
print("The number of images is %d" % imnbr)

# Create matrix to store all flattened images
immatrix = np.array([np.array(cv2.imread(imname)).flatten() for imname in imlist], 'f')

# PCA reduce dimension
V, S, immean = pca.pca(immatrix)

# Keep the mean and principal components
f = open('./a_pca_modes.pkl', 'wb')
pickle.dump(immean,f)
pickle.dump(V,f)
f.close()

# load model file
with open('./a_pca_modes.pkl','rb') as f:
    immean = pickle.load(f)
    V = pickle.load(f)

# create matrix to store all flattened images
immatrix = np.array([np.array(cv2.imread(im)).flatten() for im in imlist])
immatrix_src = np.array([np.array(cv2.imread(im)) for im in imlist])

# project on the 40 first PCs
immean = immean.flatten()
projected = np.array([dot(V[:40],immatrix[i]-immean) for i in range(imnbr)])

# k-means
projected = whiten(projected)
centroids,distortion = kmeans(projected, 10)
code,distance = vq(projected,centroids)

# plot clusters
for k in range(10):
    ind = where(code==k)[0]
    print("class:", len(ind))
    for i in range(minimum(len(ind),40)):
        cv2.imshow("123", immatrix_src[ind[i]])
        cv2.waitKey()




