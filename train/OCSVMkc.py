# -*- coding: utf-8 -*-

import numpy as np
from parm import *
from utils.utilts import *


def search():
    """
    调参
    随机搜索OCSVM的特征和训练样本你的数目
    使用不同的训练样本和相同的验证集3次取平均
    保存auc最高的模型
    :return:
    """

    # 训练误差分数的上限和支持向量分数的下限
    nus = [0.1, 0.01, 0.001, 0.0001]
    # 训练样本容忍率
    tols = [0.1, 0.01, 0.001, 0.001, 0.0001]
    # 核系数
    gammas = [1e-06, 1e-05, 1e-04, 1e-03, 1e-02, 1e-01, 0.2]
    # 是否启发式搜索
    shrinkings = [True, False]
    train_data = pd.read_csv(os.path.join(features_file, 'train_224.csv'))
    save_file = os.path.join(features_file, 'parm', 'OCSVM.csv')
    print('training')
    nums = [5000, 10000, 20000, 50000]
    max1 = 0
    scaler = StandardScaler()
    dfs = pd.DataFrame(
            columns=[
                'tol:',
                "nu:",
                'num:',
                "shrinking:",
                'gamma:',
                'auc','acc'])
    for num in nums:
        for _ in range(30):
            print(_)
            # 随机取参数
            nu = nus[np.random.randint(0, len(nus))]
            tol = tols[np.random.randint(0, len(tols))]
            gamma = gammas[np.random.randint(0, len(gammas))]
            shrinking = shrinkings[np.random.randint(0, len(shrinkings))]
            # 在train_data中抽取num个样本
            X = train_data.sample(n=num)
            det = OCSVM(nu=nu, gamma=gamma, tol=tol, shrinking=shrinking)
            pipeline = make_pipeline(scaler,det).fit(X)
            auc1 = get_auc(pipeline)
            acc1 = get_acc(pipeline)
            if max1 < acc1:
                max1 = acc1
                print("acc:",acc1,"auc:",auc1)

                print(
                    'tol:',
                    tol,
                    "nu:",
                    nu,
                    'num:',
                    num,
                    "shrinking:",
                    shrinking,
                    'gamma:',
                    gamma)

                joblib.dump(pipeline, f'../model/' +
                            str(det).split('(')[0] + ".m")

            xx = pd.DataFrame(data=[[tol, nu, num, shrinking, gamma, auc1, acc1]],
                              columns=[
                'tol:', "nu:", 'num:', "shrinking:", 'gamma:', 'auc','acc']
            )
            dfs = pd.concat([dfs, xx])
            dfs.to_csv(save_file)


if __name__ == '__main__':

    search()
