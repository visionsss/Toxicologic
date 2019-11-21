# -*- coding: utf-8 -*-
"""
@Time    : 9/18/19 11:51 AM
@Author  : vision
"""
import openslide as opsl
from sklearn.metrics import roc_curve, auc
import cv2
import re
import copy
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from glob import glob
from skimage import io
import shutil
from keras.applications import VGG16, ResNet50
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
from keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model
from keras.optimizers import SGD
import tensorflow as tf
from keras import backend as K
from kenchi.outlier_detection import IForest,LOF,OCSVM
from sklearn.externals import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score,confusion_matrix
from kenchi.pipeline import make_pipeline
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # delete warning
config = tf.ConfigProto()
config.gpu_options.allow_growth=True
sess = tf.Session(config=config)
K.set_session(sess)



##############################
#经常需要修改的参数

# 选择获取特征的模型 VGG16 | Resnet50 | Pathology
get_features_model_name = 'VGG16'
# get_features_model_name = 'Resnet50'
# get_features_model_name = 'Pathology'

# LOF、OCSVM、IForest模型路径
model_path = f'/cptjack/totem/Toxicology/{get_features_model_name}features/model/IForest.m'

# 把大图切成小图后存放的目录
# save_cut_png_dir = '/cptjack/totem/Toxicology/pictures'
save_cut_png_dir = '/cptjack/totem/Toxicology/Normalize_pictures'

# 提取特征存放csv文件位置
features_file = f'/cptjack/totem/Toxicology/{get_features_model_name}features'
################################


# vgg 512 resnet50 2048
if get_features_model_name == 'VGG16':
    features_num = 512
elif get_features_model_name == 'Resnet50':
    features_num = 2048
elif get_features_model_name == 'Pathology':
    features_num = 1024

# VGG16或Resnet50权重位置
model_weight = f'/cptjack/sys_software_bak/keras_models/models/{get_features_model_name.lower()}_weights_tf_dim_ordering_tf_kernels_notop.h5'

# 医生标注的异常部分bmp图像目录
bmp_dir = '/cptjack/totem/Toxicology/pictures/PDF_extract'

# 日本中心正常svs图数据集目录
japan_normal_dir = '/cptjack/totem/Toxicology/Set 1 Japan/Train'

# 切图大小size
size = 224

# vgg16模型输入大小,默认是224*224*3
model_size = 224

