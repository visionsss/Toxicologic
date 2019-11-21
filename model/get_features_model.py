# -*- coding: utf-8 -*-
# @Time : 9/8/19 2:08 AM
# @Author : vision
from parm import *


def rebuilt_vgg16model(size):
    """
    重建VGG16模型
    :param size: image input shape (size,size,3)
            原始vgg16无头全连接层Imagenet权重:
            vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5
    :return: vgg16model
                把最后n*n*512个特征取平均值得到512个特征
    """
    vgg16_model = VGG16(
        include_top=False,
        weights=model_weight,
        input_shape=(
            size,
            size,
            3),
        pooling='avg')
    vgg16_model.compile(
        loss='categorical_crossentropy',
        optimizer=SGD(
            lr=0.01,
            momentum=0.9,
            nesterov=True))
    vgg16_model.save(f"../model/vgg16_{size}.h5")
    print("success to creat vgg16 model")


def rebuilt_resnet50model(size):
    """
    重建Restnet50模型
    :param size: image input shape (size,size,3)
            原始Resnet50无头全连接层Imagenet权重:
            resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5
    :return: Resnet50model
                把最后n*n*2048个特征取平均值得到512个特征
    """
    resnet50_model = ResNet50(
        include_top=False,
        weights=model_weight,
        input_shape=(
            size,
            size,
            3),
        pooling='avg')
    resnet50_model.compile(
        loss='categorical_crossentropy',
        optimizer=SGD(
            lr=0.01,
            momentum=0.9,
            nesterov=True))
    resnet50_model.save(f"../model/resnet50_{size}.h5")
    print("success to creat resnet50 model")


def get_model(size=224):
    """
    :return:重建的VGG16 | Resnet50模型
    """
    if get_features_model_name == 'VGG16':
        try:
            vgg16_model = load_model(f'../model/vgg16_{size}.h5')
            print("success to get vgg16 model")
        except BaseException:
            rebuilt_vgg16model(size=size)
            vgg16_model = load_model(f'../model/vgg16_{size}.h5')
        return vgg16_model
    else:
        try:
            resnet50_model = load_model(f'../model/resnet50_{size}.h5')
            print("success to get resnet50 model")
        except BaseException:
            rebuilt_resnet50model(size=size)
            resnet50_model = load_model(f'../model/resnet50_{size}.h5')
        return resnet50_model


if __name__ == '__main__':
    model = get_model(size=model_size)
    img = np.random.uniform(0, 1, size=(10, 224, 224, 3))
    img=img.reshape(-1,224,224,3)
    y = model.predict(img)
    print(y)
    from keras.utils import plot_model
    plot_model(model, 'model.png', show_shapes=True)
