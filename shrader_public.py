import cv2
import os
import numpy as np

Xa = []
Xb = []
y = []

IM_DIR = "/media/yonatan/magnetic/PycharmProjects/DLcourse/images2/"
OUTPUT_DIR = "output/"
files = os.listdir(IM_DIR)

# update this number for 4X4 crop 2X2 or 5X5 crops.
tiles_per_dim = 4



for f in files:
    im = cv2.imread(IM_DIR+f)
    im = cv2.cvtColor(im,cv2.COLOR_RGB2GRAY)
    height = im.shape[0]
    width = im.shape[1]
    frac_h = height//tiles_per_dim
    frac_w = width//tiles_per_dim
    i=0
    for h in range(tiles_per_dim):
        for w in range(tiles_per_dim):

            crop = im[h*frac_h:(h+1)*frac_h,w*frac_w:(w+1)*frac_w]
            cv2.imwrite(OUTPUT_DIR+f[:-4]+"_{}.jpg".format(i),crop)
            i=i+1









