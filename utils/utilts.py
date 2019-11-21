# -*- coding: utf-8 -*-
from parm import *


def read_roc_csv():
    """
    读取测试集
    :return:前半部分正常图，后半部分异常图
    """
    test_path = os.path.join(features_file, 'test_224.csv')
    data = pd.read_csv(test_path).iloc[:, :].values
    x = data[:, :-1]
    y = data[:, -1]
    return x, y


def get_auc(model):
    """
    获取模型auc值
    :param model: 是训练好的(LOF)模型
    :return:  roc曲线下auc的值
    """
    x, y = read_roc_csv()
    dis = -model.decision_function(x)
    false_positive_rate, true_positive_rate, thresholds = roc_curve(
        y, dis, pos_label=1)
    roc_auc = auc(false_positive_rate, true_positive_rate)
    return roc_auc
def plot_roc(model):
    """
    draw roc曲线
    :param model: 是训练好的(LOF)模型
    draw roc曲线下auc的值
    """
    x, y = read_roc_csv()
    for i in range(len(y)):
        if(y[i]==1):
            y[i]=-1
        else:
            y[i]=1
    model.plot_roc_curve(x, y)
    plt.show()
def get_acc(model):
    """
    获取模型acc值
    :param model: 是训练好的(LOF)模型
    :return: acc的值
    """
    x, y = read_roc_csv()
    pre = model.predict(x)
    for i in range(len(pre)):
        if (pre[i] == 1):
            pre[i] = 0
        else:
            pre[i] = 1
    acc = accuracy_score(y, pre)
    cm = confusion_matrix(y, pre)
    plt.imshow(cm, interpolation='nearest', cmap='Pastel1')
    plt.title('Confusion matrix', size=15)
    plt.colorbar()
    tick_marks = np.arange(2)
    plt.xticks(tick_marks, ['normal', 'abnormal'], rotation=45, size=10)
    plt.yticks(tick_marks, ['normal', 'abnormal'], size=10)
    plt.tight_layout()
    plt.ylabel('Actural label', size=15)
    plt.xlabel('Predicted label', size=15)
    width, height = cm.shape
    for x in range(width):
        for y in range(height):
            plt.annotate(str(cm[x][y]), xy=(y, x))
    #plt.show()
    return acc


def get_steps(slide, size=224):
    """
    获取能横、竖切多少刀
    :param slide: openslide读取的svs图
    :param size:
    :return:
    """
    w_count = slide.level_dimensions[0][0] // size
    h_count = slide.level_dimensions[0][1] // size
    return w_count, h_count


def judge(im, thr=0.90):
    """
    判断图片是否丢弃
    :param im:输入图片为RGB格式的图片(数组)
    :param thr: 白色区域大于thr则不要该图片
    :return:图片白色区域小于thr返回1，其他返回0
            (即判断图片是否含有细胞)
    """
    hsv = cv2.cvtColor(im, cv2.COLOR_RGB2HSV)
    lower = np.array([0, 0, 0])
    upper = np.array([256, 19, 256])
    mask = cv2.inRange(hsv, lower, upper)
    if np.sum(mask > 0) / (im.shape[0] * 
             im.shape[1]) < thr:
        return 1
    else:
        return 0



def show2(model):
    """
    显示模型在测试集的效果
    :param model: 是训练好的(LOF)模型
    """
    x, y = read_roc_csv()
    # arr = model.predict_proba(x)[:,0]
    arr = model.decision_function(x)
    arr1 = arr[:len(arr) // 2]
    arr2 = arr[len(arr) // 2:]
    plt.rcParams['figure.figsize'] = 4, 4
    n, bins, patches = plt.hist(arr1, bins=50, color='b')

    n, bins2, patches = plt.hist(arr2, bins=50, color='r', alpha=0.5)
    plt.legend(['normal', 'abnormal'])
    plt.show()
    
    
def imagenet_processing(img):
    """
    图片预处理
    :parm img 是图像矩阵
    ：return 经过减去均值方差之后的图像矩阵
    """
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    for i in range(3):
        img[:, :, i] -= mean[i]
        img[:, :, i] /= std[i]
    return img


def get_mpp(slide):
    """
    获取slide图片(svs大图)的倍数
    :param slide: opsl.OpenSlide(svs_path)
    :return: 获取slide图片mpp值
    """
    return np.float(slide.properties['openslide.mpp-x'])


if __name__ == '__main__':
    model = joblib.load(model_path)
    show2(model)
    auc_ = get_auc(model)
    print('auc: ',auc_)
    acc_ = get_acc(model)
    print('acc: ', acc_)
    plot_roc(model)
