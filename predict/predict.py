# -*- coding: utf-8 -*-
from utils.utilts import *
from parm import *
import time
from model.get_features_model import get_model
from utils.stain_trans import standard_transfrom


def get_features(svs_path):
    """
    获取svs图片的特征;并保存在features_file(parm参数) + /test_svs路径上
    :parm svs_path svs图路径
    :return count_list 每个patch的x，y坐标 shape=(-1，2)
    :return feature 每个patch的特征 shape=(-1，512)
    """
    # 存储特征
    features = []
    # 存储位置
    count_list = []
    ###################
    #先查找features_file + /test_svs文件下有无该svs图的特征
    #有就直接保存
    #没有就切图->vgg16提取特征
    ##################
    had = glob(os.path.join(features_file, 'test_svs', '*.csv'))
    flag = 0
    svs_num = re.findall("/([0-9]+).svs", svs_path)[0]
    for hadd in had:
        dir_num = re.findall("/([0-9]+).csv", hadd)[0]
        if dir_num == svs_num:# 如果找到存在的文件就打断
            flag = 1
            break
    if flag:
        #"该svs图像已经提取过特征保存在features_file + /test_svs文件下
        file_path = os.path.join(features_file, 'test_svs', f'{svs_num}.csv')
        test = pd.read_csv(file_path)
        count_list = test.iloc[:, 0:2].values
        features = test.iloc[:, 2:].values

    else:
        # svs图像没有提取过特征，现在开始切图并提取特征
        model = get_model()
        slide = opsl.open_slide(svs_path)
        w_count, h_count = get_steps(slide=slide, size=size)
        count = 0
        standard_img = io.imread('../normalization/standard_img.png')
        stain_method_M = standard_transfrom(standard_img=standard_img, method='M')
        for w in range(w_count):
            for h in range(h_count):
                slide_region = np.array(slide.read_region((w * size, h * size), 0,
                                                          (size, size)))[:, :, 0:3]
                # 判断该图片是否是空白图片
                if judge(slide_region, thr=0.15):
                    # 图像预处理
                    # 标准化
                    """
                    try:
                        slide_region = stain_method_M.transform(slide_region)
                    except:
                        # 几乎全白的图片才会出现异常
                        print(count, ' error')
                    """
                    x = slide_region/255.0
                    x = imagenet_processing(x)
                    
                    # 提取特征
                    x = np.expand_dims(x, axis=0)
                    y = model.predict(x)
                    features.append(y)
                    count_list.append(np.array([str(w * size), str(h * size)]))
                count += 1
                if count % 1000 == 0:
                    # 打印切图和提取特征进度
                    print(count, ' / ', w_count * h_count)

        # 保存特征为csv文件
        features = np.array(features).reshape(-1, features_num)
        count_list = np.array(count_list).reshape(-1, 2)
        all_f = np.append(count_list, features, axis=1)
        df = pd.DataFrame(all_f)
        file_path = os.path.join(features_file, 'test_svs', f'{svs_num}.csv')
        df.to_csv((file_path), header=True, index=False)
    return count_list, features


def predict_proba(features):
    """
    预测概率
    :parm features 一张svs图的特征
    :return : 每一个patch异常的概率
    """
    model = joblib.load(model_path)
    proba = model.predict_proba(features)[:, 0]
    return proba


def draw(svs_path, proba, count_list):
    """
    画热图
    :parm svs_path是svs图片的路径
    :parm proba 每一个patch异常的概率
    :parm cout_list 每一个patch的位置信息，由get_features返回
    """
    # 在svs_path中寻找svs_num(如26761)
    svs_num = re.findall("/([0-9]+).svs", svs_path)[0]
    slide = opsl.OpenSlide(svs_path)
    # 获取svs图它能切多少刀
    w_count, h_count = get_steps(slide)

    # 保存原始svs略缩图
    slide_thumbnail = slide.get_thumbnail(slide.level_dimensions[2])
    slide = np.array(slide_thumbnail)
    slide = cv2.resize(slide,dsize=(slide.shape[1]//10,slide.shape[0]//10))
    io.imsave(f"../pictures/o{svs_num}.png", slide)

    # 画热力图
    # x ,y 存放热力图图像的x，y坐标
    x = []
    y = []
    for i in range(len(count_list)):
        ccc = count_list[i] # count_list 存放原始图像的x，y坐标,so need //224
        h = int(ccc[0]) // 224
        w = int(ccc[1]) // 224
        x.append(h)
        y.append(w_count-w) # scatter y轴坐标需反过来，不然画出来的图是反的
    # 把x，y根据proba异常概率点上去
    fig, ax = plt.subplots(figsize=(w_count / 50, h_count / 50))
    plt.axis('off')
    im = plt.scatter(x, y, c=proba, s=0.1, cmap='jet')
    plt.colorbar(im)
    plt.title('abnormal: ' + str(round(np.sum(proba > 0.5) / len(proba) * 100, 2)) + '%')
    if not os.path.exists("../pictures"):
        os.mkdir('../pictures')
    plt.savefig(f"../pictures/{svs_num}.png")


def predict(svs_path):
    """
    预测一张svs大图病变patch的结果
    :parm 保存在Toxicology/pictures/hot、black、original分别对应热图、预测部位图、原图
    """
    
    t1 = time.time()
    print("get_features")
    count_list, features = get_features(svs_path)
    t2 = time.time()
    print('predict')
    proba = predict_proba(features)
    t3 = time.time()
    print('draw')
    draw(svs_path, proba, count_list)
    t4 = time.time()
    print(" get_features: ", t2 - t1)
    print("predict_proba: ", t3 - t2)
    print("         draw: ", t4 - t3)
    return proba


if __name__ == '__main__':


    # Japan
    #     normal
    svs_path = '/cptjack/totem/Toxicology/Set 1 Japan/Test/1-acetaminophen/liver/26761.svs'
    proba1 = predict(svs_path)
    #     Moderate
    svs_path = '/cptjack/totem/Toxicology/Set 1 Japan/Test/Moderate and severe cases/Liver/moderate/27237.svs'
    proba2 = predict(svs_path)
    #     severe
    svs_path = '/cptjack/totem/Toxicology/Set 1 Japan/Test/Moderate and severe cases/Liver/severe/29550.svs'
    proba1 = predict(svs_path)










