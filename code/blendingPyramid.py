# -*- coding: utf-8 -*-
"""
Created on Fri Oct 07 14:59:50 2016

@author: zhouc
"""
import math
import numpy as np
import cv2

def createPyramid(img, pyramidN):
    imagePyramid = list()
    gaussianPyramid = list()
    laplacePyramid = list()

    imagePyramid.append(img);
    image = img.copy();
    for i in range(pyramidN):
        tmp_img = cv2.resize(image,(image.shape[0]/2,image.shape[1]/2),interpolation = cv2.INTER_CUBIC)
        imagePyramid.append(tmp_img);
        tmp_img = cv2.resize(tmp_img,(image.shape[0], image.shape[1]),interpolation = cv2.INTER_CUBIC)
        gaussianPyramid.append(tmp_img);
        laplacePyramid.append(image - tmp_img);
        image = imagePyramid[i+1];

    return imagePyramid, gaussianPyramid, laplacePyramid

# config & input
Topic = 'lion'

backImageName = Topic + '.png'
foreImageName = Topic + '2.png'
maskName = Topic + '_mask.png'
outputName = Topic + '_out_pyramid.png'

backImg = cv2.imread('./pyramid/' + backImageName) / 255.0
foreImg = cv2.imread('./pyramid/' + foreImageName) / 255.0
mask = cv2.imread('./pyramid/' + maskName) / 255.0

rows = backImg.shape[0]
cols = backImg.shape[1]
channels = backImg.shape[2]

if mask.ndim == 2:
    mask = np.reshape(mask, [mask.shape[0], mask.shape[1], 1])

if mask.shape[2] == 1:
    mask = np.tile(mask, [1, 1, 3])

pyramidN = int(math.ceil(math.log(min(rows, cols) / 16, 2)));


# build pyramid
[imageFore, gaussianFore, laplaceFore] = createPyramid(foreImg, pyramidN)
[imageBack, gaussianBack, laplaceBack] = createPyramid(backImg, pyramidN)
[imageMask, gaussianMask, laplaceMask] = createPyramid(mask, pyramidN)

# combine laplacian pyramid
laplaceMerge = list()
"""
TODO 2
Combine the laplacian pyramids of background and foreground
add your code here
"""
for i in range(pyramidN):
    k = laplaceFore[i]*gaussianMask[i]+(1-gaussianMask[i])*laplaceBack[i]
    laplaceMerge.append(k)
# Combine the smallest scale image
"""
TODO 3
Combine the smallest scale images of background and foreground

add your code here
"""
smallestImg = imageFore[pyramidN]*imageMask[pyramidN] + (1-imageMask[pyramidN])*imageBack[pyramidN]
# reconstruct & output
"""
TODO 4
reconstruct the blending image by adding the gradient (in different scale) back to
the smallest scale image while upsampling
add your code here
"""
for i in range(1,pyramidN+1):
    smallestImg = cv2.resize(smallestImg, (laplaceMerge[pyramidN-i].shape[0], laplaceMerge[pyramidN-i].shape[1]))
    smallestImg = smallestImg + laplaceMerge[pyramidN-i];

img = smallestImg;

cv2.imshow('output', img);
cv2.waitKey(0)
cv2.imwrite(outputName, img * 255);
