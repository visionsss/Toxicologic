from parm import *
from utils.stain_trans import standard_transfrom
"""
origin_dir = '/cptjack/totem/Toxicology/pictures'
normalize_dir = '/cptjack/totem/Toxicology/Normalize_pictures'
将origin_dir内所有png图片进行标准化,保存至normalize_dir中
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


def normalize_pictures(dir_origin):
    """
    将原来的图片进行标准化后保存
    原来图片/cptjack/totem/Toxicology/pictures/224
    标准化后保存/cptjack/totem/Toxicology/Normalize_pictures/224
    :param dir_origin:原始图片路径
    :return:
    """
    # 获取该目录下所有png图片
    dir_origin = get_png_path(os.path.join(dir_origin,'*'))
    # 获取标准化模型
    standard_img = io.imread('standard_img.png')
    stain_method_M = standard_transfrom(standard_img=standard_img, method='M')
    # 对每一张图片进行标准化，标准化失败则保存原来的图片
    for i in range(len(dir_origin)):
        save_path = dir_origin[i].replace('pictures','Normalize_pictures')
        img = io.imread(dir_origin[i])
        try:
            img = stain_method_M.transform(img)
        except:
            # 几乎全白的图片才会出现异常
            print(i,' error')
        io.imsave(save_path, img)
        # 打印进度
        if i%1000 == 0:
            print(i)


if __name__ == '__main__':
    # 未标准化的图片保存目录
    origin_dir = '/cptjack/totem/Toxicology/pictures'
    # 标准化的图片保存目录
    normalize_dir = '/cptjack/totem/Toxicology/Normalize_pictures'
    # 获取所有需要标准化的文件目录
    x = glob(os.path.join(origin_dir,'*224'))
    for i in range(0, len(x)):
        # 进行标准化并保存图片
        normalize_pictures(x[i])