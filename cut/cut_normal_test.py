# -*- coding: utf-8 -*-
"""
@Time    : 9/18/19 2:14 PM
@Author  : vision
"""
from utils.utilts import judge, get_steps
from parm import *
def get_train_svs_path():
    """
    get_all_svs_path
    """
    dir1 = os.path.join(japan_normal_dir, 'isoniazid', '*')
    dir2 = os.path.join(japan_normal_dir, 'valproic', '*')
    file_list = glob(dir1)
    file_list.extend(glob(dir2))
    np.random.seed(123)
    np.random.shuffle(file_list,)
    return file_list


def cut_224(svs_path, size):
    """
    添加切割正常图方法
    :param svs_path: *.svs
    :param size: in parm.py
    :return:
    """
    global count
    slide = opsl.open_slide(svs_path)
    tmp_path = os.path.join(save_cut_png_dir, 'tmp')
    if not os.path.exists(tmp_path):
      os.makedirs(tmp_path)
    w_count, h_count = get_steps(slide, size=size)
    for h in range(10, h_count):
        for w in range(10, w_count):
            slide_region = np.array(slide.read_region((w * size, h * size), 0,
                                                      (size, size)))[:, :, 0:3]
            # 判断该图片是否是空白图片

            if np.random.random() < 0.15:
                if judge(slide_region):
                    io.imsave(os.path.join(tmp_path, f'{count}.png'), slide_region)
                    count+=1
    return tmp_path


def select(num, origin_file, to_file):
    """
    :param num:
    :param origin_file:
    :param to_file:
    :return:
    """
    origin_file = os.path.join(origin_file, '*')
    to_file = to_file + r'/'
    file_path = glob(origin_file)
    np.random.shuffle(file_path)
    for i in range(len(file_path)):
        shutil.move(file_path[i], to_file + str(i) + '.png')
        print(file_path[i], to_file + str(i) + '.png')
        if i == num-1:
            break

if __name__ == '__main__':
    data_file = get_train_svs_path()[71:]

    count = 0
    for kk in range(len(data_file)):
        tmp_path = cut_224(svs_path=data_file[kk], size=size)
        print(kk)
    select(6103, tmp_path, os.path.join(save_cut_png_dir, 'test_224'))
