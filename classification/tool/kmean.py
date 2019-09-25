# -*- coding: UTF-8 -*-
import cv2
import os
import numpy as np
import sys
import pickle
from PCV.tools import imtools
from scipy import *
from pylab import *
from scipy.cluster.vq import *
from PCV.tools import pca

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

os.mkdir('./kmeans_result')
# plot clusters
for k in range(10):
    ind = where(code==k)[0]
    print("class:",  k, len(ind))
    os.mkdir('./kmeans_result/' + str(k))
    for i in range(len(ind)):
        cv2.imwrite('./kmeans_result/' + str(k) + '/' + str(i) + '.jpg', immatrix_src[ind[i]])




