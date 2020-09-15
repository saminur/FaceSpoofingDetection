# -*- coding: utf-8 -*-
"""
Created on Thu Apr 23 16:42:47 2020

@author: sami
"""


#from skimage import io as skio

import numpy as np 

#from matplotlib import pyplot as plt 
import cv2
from PIL import Image 
import os

def cv2pil(cv_im):
    # Convert the cv image to a PIL image
    return Image.fromstring("L", cv2.GetSize(cv_im), cv_im.tostring())

def imgCrop(image, cropBox, boxScale=1):
    # Crop a PIL image with the provided box [x(left), y(upper), w(width), h(height)]

    # Calculate scale factors
    xDelta=max(cropBox[2]*(boxScale-1),0)
    yDelta=max(cropBox[3]*(boxScale-1),0)

    # Convert cv box to PIL box [left, upper, right, lower]
    PIL_box=[cropBox[0]-xDelta, cropBox[1]-yDelta, cropBox[0]+cropBox[2]+xDelta, cropBox[1]+cropBox[3]+yDelta]

    return image.crop(PIL_box)


casePath="C:\\Users\\samin\\Anaconda3\\Lib\\site-packages\\cv2\\data\\haarcascade_frontalface_default.xml" 
# need to put their installed library location
face_cascade = cv2.CascadeClassifier(casePath)
face_cascade.load(casePath)

 
train_path       = "train"
train_labels = os.listdir(train_path)
saved_path="train_cropped"
# sort the training labels
train_labels.sort()
print(train_labels)
fixed_size       = tuple((224, 224))  
for training_name in train_labels:
    dir = os.path.join(train_path, training_name)
    for file in os.listdir(dir):
        img = cv2.imread(os.path.join(dir, file))
 
        image = cv2.resize(img, fixed_size) 
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        mask = np.zeros(image.shape[:2], np.uint8) 
        backgroundModel = np.zeros((1, 65), np.float64) 
        foregroundModel = np.zeros((1, 65), np.float64) 
        faces = face_cascade.detectMultiScale(gray, 2, 5)
        fname=saved_path+training_name+"\\"+file
        pil_im=Image.open(os.path.join(dir, file))
        
        if len(faces)==0:
            rectangle = (16,24,192,192)
            cv2.grabCut(image, mask, rectangle, backgroundModel, foregroundModel, 3, cv2.GC_INIT_WITH_RECT) 
            mask2 = np.where((mask == 2)|(mask == 0), 0, 1).astype('uint8') 
            image = image * mask2[:, :, np.newaxis] 
            cv2.imwrite(fname, image)
                        
        else:
            rectangle = faces[0]
            cv2.grabCut(image, mask, rectangle, backgroundModel, foregroundModel, 3, cv2.GC_INIT_WITH_RECT) 
            mask2 = np.where((mask == 2)|(mask == 0), 0, 1).astype('uint8') 
            image = image * mask2[:, :, np.newaxis] 
            cv2.imwrite(fname, image) 

        
        