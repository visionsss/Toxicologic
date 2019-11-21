# -*- coding: utf-8 -*-
"""
@Time    : 9/19/19 9:51 AM
@Author  : vision
"""
from parm import *
from model.get_features_model import get_model
from utils.utilts import *
"""
得到测试集合的x和y保存至test_224.csv
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


def get_feature(normal_path, abnormal_path, size):
    """
    将test_224和abnormal_224内全部突破提取特征存放在test_224.csv中
    :parm normal_path 切好的6104张正常图路径
    :parm abnormal_path 切好的6104张异常图路径
    :parm size 模型大小
    """
    # save features
    features = []
    # save label 正常0，异常1
    ys = []
    model = get_model(size=size)
    # 获取正常的png图片的features和label
    file_path = get_png_path(normal_path)
    for i in range(len(file_path)):
        if i % 1000 == 0:
            print(i)
        # 图片预处理
        img = image.load_img(file_path[i], target_size=(224, 224))
        x = image.img_to_array(img)
        x = x/255.0
        x = imagenet_processing(x)
        x = np.expand_dims(x, axis=0)
        # 获取特征和标签
        feature = model.predict(x)
        features.append(feature)
        ys.append(0)
    # 获取异常的png图片的features和label
    file_path = get_png_path(abnormal_path)
    for i in range(len(file_path)):
        if i % 1000 == 0:
            print(i)
        # 图片预处理
        img = image.load_img(file_path[i], target_size=(224, 224))
        x = image.img_to_array(img)
        x = x/255.0
        x = imagenet_processing(x)
        x = np.expand_dims(x, axis=0)
        # 获取特征和标签
        feature = model.predict(x)
        features.append(feature)
        ys.append(1)

    # 将features和ys合并，保存至test_224.csv
    ys = np.array(ys).reshape(-1, 1)
    features = np.array(features).reshape(-1, features_num)
    ys = pd.DataFrame(ys)
    features = pd.DataFrame(features)
    print(ys.shape)
    print(features.shape)
    all_ = pd.concat([features, pd.DataFrame(ys)], axis=1)
    file_names = os.path.join(features_file, 'test_224.csv')
    all_.to_csv((file_names), header=True, index=False)





if __name__ == '__main__':
    get_feature(
        normal_path=os.path.join(save_cut_png_dir, 'test_224', '*'),
        abnormal_path=os.path.join(save_cut_png_dir, 'abnormal_224', '*'),
        size=model_size
    )
