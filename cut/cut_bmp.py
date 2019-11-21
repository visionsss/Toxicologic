# -*- coding: utf-8 -*-
"""
@Time    : 9/18/19 12:00 PM
@Author  : vision
"""
from skimage import io
from glob import glob
import os
from parm import *


def cut(bmp_dir, save_cut_png_dir, size):
    """
    切割医生标注的异常bmp图片
    :param bmp_dir:参数在parm.py文件中,bmp的上级文件夹目录，默认/cptjack/totem/Toxicology/abnormal/PDF_extract
            文件内有很多个文件夹，每个文件夹下面有异常的bmp图,
    :param save_cut_png_dir: 参数在parm.py文件中,把大图切成小图后存放的目录
    :param size:参数在parm.py文件中,切割图片的大小(size，size，3)
    :return:
    """
    # 异常图存放目录abnormal_224
    save_dir = os.path.join(save_cut_png_dir, 'abnormal_224')
    count = 0
    # 获取文件目录下所有文件夹
    for i in glob(os.path.join(bmp_dir, '*')):
        # 获取每个文件夹目录下的bmp图片
        k = sorted(glob(i + '/*.bmp'))
        for j in k:
            # read .bmp
            a = io.imread(j)
            # cut .bmp
            for r in range(a.shape[0] // size):
                for c in range(a.shape[1] // size):
                    new_pic = a[r * size:(r + 1) * size, c * size:(c + 1) * size]

                    files = os.path.join(save_dir, f'{count}.png')
                    io.imsave(files, new_pic)
                    count += 1


if __name__ == '__main__':
    cut(bmp_dir, save_cut_png_dir, size)

