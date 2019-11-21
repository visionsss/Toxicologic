# -*- coding: utf-8 -*-

import numpy as np
from parm import *
from utils.utilts import *


def search():
    """
    调参
    随机搜索OLOF的特征和训练样本你的数目
    使用不同的训练样本和相同的验证集3次取平均
    保存auc最高的模型
    """
    leaf_sizes = [10, 20, 30, 50, 80, 100]
    algorithms = ['auto', 'ball_tree', 'kd_tree', 'brute']
    contaminations = [0.001,0.005, 0.01, 0.02, 0.05,0.1]
    n_neighborss = [5, 10, 20, 30, 40, 50, 80, 100]
    train_data = pd.read_csv(os.path.join(features_file, 'train_224.csv'))
    save_file = os.path.join(features_file, 'parm', 'LOF.csv')
    print('training')
    scaler = StandardScaler()
    nums = [5000, 10000]
    max1 = 0
    dfs = pd.DataFrame(
            columns=[
                'algorithm:',
                "leaf_size",
                'num:',
                "contamination",
                'n_neighbors',
                'auc','acc'])
    for num in nums:
        for _ in range(20):
            print(_)
            algorithm = algorithms[np.random.randint(0, len(algorithms))]
            leaf_size = leaf_sizes[np.random.randint(0, len(leaf_sizes))]
            contamination = contaminations[np.random.randint(
                0, len(contaminations))]
            n_neighbors = n_neighborss[np.random.randint(0, len(n_neighborss))]

            X = train_data.sample(n=num)
            det = LOF(novelty=True, algorithm=algorithm, leaf_size=leaf_size,
                      contamination=contamination,
                      n_neighbors=n_neighbors, n_jobs=5)
            pipeline = make_pipeline(det).fit(X)
            # save model
            auc1 = get_auc(pipeline)
            acc1 = get_acc(pipeline)
            if max1 < acc1:
                max1 = acc1
                print("acc:",acc1,"auc:",auc1)
                joblib.dump(pipeline, '../model/' + str(det).split('(')[0] + ".m")
                print('algorithm:', algorithm,
                  "leaf_size:", leaf_size,
                  'contamination:', contamination,
                  "n_neighbors:", n_neighbors,
                  'num:', num)
            xx = pd.DataFrame(data=[[algorithm, leaf_size, num, contamination, n_neighbors, auc1, acc1]],
                              columns=[
                'algorithm:',
                "leaf_size",
                'num:',
                "contamination",
                'n_neighbors',
                'auc','acc']
            )
            dfs = pd.concat([dfs, xx])
            dfs.to_csv(save_file)


if __name__ == '__main__':
    search()
