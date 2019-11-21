from parm import *
from utils.utilts import get_mpp


if __name__ == '__main__':
    # Japan
    dirs1 = ['/cptjack/totem/Toxicology/Set 1 Japan/Train/isoniazid'
            , '/cptjack/totem/Toxicology/Set 1 Japan/Train/valproic'
            , '/cptjack/totem/Toxicology/Set 1 Japan/Test/Moderate and severe cases/Liver/moderate'
            , '/cptjack/totem/Toxicology/Set 1 Japan/Test/Moderate and severe cases/Liver/severe'
            ]
    # China
    dirs2 = [
        '/cptjack/totem/Toxicology/Set 2 China/train/normal',
        '/cptjack/totem/Toxicology/Set 2 China/test/normal',
        '/cptjack/totem/Toxicology/Set 2 China/test/abnormal'
    ]
    dir3 = [
        '/cptjack/totem/Toxicology/severe'
    ]
    for i in dir3:
        # 遍历每一张svs图
        path = os.path.join(i,'*')
        paths = glob(path)
        # 存储每一张svs图的mpp值
        mpp_count = []
        level = []
        for svs_path in paths:
            try:
                slide = opsl.OpenSlide(svs_path)
                level.append(slide.level_count)
                mpp = get_mpp(slide)
                # print(i.split('/')[-1], svs_num,": ",mpp)
                mpp_count.append(mpp)
            except:
                print(svs_path)
                pass
        level = pd.Series(level,name='level')
        mpp_count = pd.Series(mpp_count,name='mpp')
        print('文件', i)
        print(mpp_count.value_counts())
        print()



