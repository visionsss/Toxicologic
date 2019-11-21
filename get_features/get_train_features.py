# -*- coding: utf-8 -*-
"""
@Time    : 9/19/19 11:16 AM
@Author  : vision
"""
from model.get_features_model import get_model
from parm import *
from utils.utilts import *
"""
得到训练集特征保存至train_224.csv
"""


def get_png_path(png_dir):
    """
    :parm png_dir 文件夹路径
    :return file_path 该文件夹内所有png图像路径
    """
    file_path = glob(png_dir)
    file_path = sorted(
        file_path,
        key=lambda x: int(
            re.findall(
                '/([0-9]*).png',
                x)[0]))
    return file_path


def get_feature(file_path, size=224):
    """
    将224内全部突破提取特征存放在train_224.csv中
    :parm file_path 切好的20W张正常图路径
    :parm size 模型大小
    """
    # save features
    features = []
    model = get_model(size=size)

    for i in range(len(file_path)):
        if i % 1000 == 0:
            print(i)
        # 图片预处理
        img = image.load_img(file_path[i], target_size=(size, size))
        x = image.img_to_array(img)
        x = x/255.0
        x = imagenet_processing(x)
        x = np.expand_dims(x, axis=0)
        # 获取特征
        feature = model.predict(x)
        features.append(feature)

    # 将features保存至train_224.csv
    features = np.array(features).reshape(-1, features_num)
    features = pd.DataFrame(features)
    file_names = os.path.join(features_file, 'train_224.csv')
    features.to_csv((file_names), header=True, index=False)
    return features

if __name__ == '__main__':
    # 获取训练样本切好的图的路径
    file_path = os.path.join(save_cut_png_dir, '224', '*')
    file_path = get_png_path(file_path)
    # 获取训练样本特征
    features = get_feature(file_path, size=model_size)
