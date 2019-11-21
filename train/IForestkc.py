# -*- coding: utf-8 -*-
from parm import *
from utils.utilts import *


def search():
    """
    调参
    随机搜索IForest的特征和训练样本你的数目
    使用不同的训练样本和相同的验证集3次取平均
    保存auc最高的模型
    :return:
    """
    scaler = StandardScaler()
    # 总的正常样本特征集合
    train_data = pd.read_csv(os.path.join(features_file, 'train_224.csv'))
    # 森林中树的个数
    n_estimatorss = [10, 50, 100, 150, 200, 300]
    # 样本容忍度
    contaminations = [0.1, 0.05, 0.01, 0.005, 0.001]
    # 每棵树的最大样本数
    max_sampless = [512, 1024, 2048, 4096]
    # 每棵树的最大维度(VGG16最大为512)
    max_featuress = [32, 64, 128, 256, 512]
    # 是否加速搜索
    bootstraps = [True, False]
    # 训练样本数
    nums = [5000, 10000, 20000, 50000, 100000, 200000]
    save_file = os.path.join(features_file, 'parm', 'IForest.csv')
    print(save_file)
    dfs = pd.DataFrame(columns=['n_estimators', 'contamination', 'max_samples',
                                'max_features', 'num', 'bootstrap', 'auc', 'acc'])
    max1 = 0
    for num in nums:
        for _ in range(30):
            print(_)
            # 随机取参数
            n_estimators = n_estimatorss[np.random.randint(
                0, len(n_estimatorss))]
            contamination = contaminations[np.random.randint(
                0, len(contaminations))]
            max_samples = max_sampless[np.random.randint(0, len(max_sampless))]
            max_features = max_featuress[np.random.randint(
                0, len(max_featuress))]
            # num = nums[np.random.randint(0, len(nums)-3)]
            bootstrap = bootstraps[np.random.randint(0, len(bootstraps))]
            # 在train_data中抽取num个样本
            X = train_data.sample(n=num)
            det = IForest(
                n_estimators=n_estimators,
                contamination=contamination,
                max_samples=max_samples,
                max_features=max_features,
                bootstrap=bootstrap,
                n_jobs=5)
            pipeline = make_pipeline(det).fit(X)
            auc1 = get_auc(pipeline)
            acc1 = get_acc(pipeline)
            if max1 < acc1:
                max1 = acc1
                joblib.dump(pipeline, '../model/' +
                            str(det).split('(')[0] + ".m")
                print("acc:",acc1,"auc:",auc1)
                print(
                    'n_estimators:',
                    n_estimators,
                    'contamination:',
                    contamination,
                    'max_samples:',
                    max_samples,
                    'max_features',
                    max_features,
                    'num: ',
                    num,
                    'bootstrap:',
                    bootstrap)
            xx = pd.DataFrame(data=[[n_estimators, contamination, max_samples,
                                     max_features, num, bootstrap, auc1, acc1]],
                              columns=['n_estimators', 'contamination', 'max_samples',
                                       'max_features', 'num', 'bootstrap', 'auc', 'acc']
                              )
            dfs = pd.concat([dfs, xx])
            dfs.to_csv(save_file)


if __name__ == '__main__':

    search()
