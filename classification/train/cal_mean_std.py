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
