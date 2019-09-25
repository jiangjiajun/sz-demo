# -*- coding: UTF-8 -*-

#import imtools
from PCV.tools import imtools
import pickle
from scipy import *
from pylab import *
from PIL import Image
from scipy.cluster.vq import *
from PCV.tools import pca
import numpy as np


# Uses sparse pca codepath.
imlist = imtools.get_imlist('data/train/2/0/')

# 获取图像列表和他们的尺寸
im = np.array(Image.open(imlist[0]))  # open one image to get the size
m, n = im.shape[:2]  # get the size of the images
imnbr = len(imlist)  # get the number of images
print "The number of images is %d" % imnbr

# Create matrix to store all flattened images
immatrix = np.array([np.array(Image.open(imname)).flatten() for imname in imlist], 'f')

# PCA降维
V, S, immean = pca.pca(immatrix)

# 保存均值和主成分
#f = open('./a_pca_modes.pkl', 'wb')
f = open('data/train/a_pca_modes.pkl', 'wb')
pickle.dump(immean,f)
pickle.dump(V,f)
f.close()


# get list of images
imlist = imtools.get_imlist('data/train/2/0/')
imnbr = len(imlist)

# load model file
with open('data/train/a_pca_modes.pkl','rb') as f:
    immean = pickle.load(f)
    V = pickle.load(f)
# create matrix to store all flattened images
immatrix = np.array([np.array(Image.open(im)).flatten() for im in imlist],'f')

# project on the 40 first PCs
immean = immean.flatten()
projected = np.array([dot(V[:40],immatrix[i]-immean) for i in range(imnbr)])

# k-means
projected = whiten(projected)
centroids,distortion = kmeans(projected,4)
code,distance = vq(projected,centroids)

# plot clusters
for k in range(4):
    ind = where(code==k)[0]
    #plt.figure()
    #plt.gray()
    for i in range(minimum(len(ind),40)):
        #plt.subplot(4,10,i+1)
        immatrix[ind[i]].reshape((25,25)).save("./" + str(i) + ".jpg")
        #plt.imshow(immatrix[ind[i]].reshape((25,25)))
        #plt.axis('off')

