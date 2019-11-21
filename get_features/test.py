# -*- coding: utf-8 -*-
"""
@Time    : 9/23/19 4:30 AM
@Author  : vision
"""

from parm import *


def imagenet_processing(x):
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    for i in range(3):
        x[:,:,i] -= mean[i]
        x[:,:,i] /= std[i]
    return x


png = '/cptjack/totem/Toxicology/VGG16data_set/Japan/test_224/0.png'
img = image.load_img(png, target_size=(224, 224))
xx = image.img_to_array(img).reshape(1,224,224,3)
xx1 = preprocess_input(xx)
xx1 = xx1.reshape(224,224,3)
print(xx1)


img = image.load_img(png, target_size=(224, 224))
xx = image.img_to_array(img)
xx = xx/255
y = imagenet_processing(xx).reshape(224,224,3)
print(y)









