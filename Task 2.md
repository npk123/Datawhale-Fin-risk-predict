# Datawhale-Fin-risk-predict task 2
# 目的：
1.EDA价值主要在于熟悉了解整个数据集的基本情况（缺失值，异常值），对数据集进行验证是否可以进行接下来的机器学习或者深度学习建模.
2.了解变量间的相互关系、变量与预测值之间的存在关系。
3.为特征工程做准备
# 导入数据分析及可视化过程需要的库：

    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    import datetime
    import warnings
    warnings.filterwarnings('ignore')
# 读取文件
    data_train = pd.read_csv('./train.csv')
    data_test_a = pd.read_csv('./testA.csv')
    data_train_sample = pd.read_csv("./train.csv",nrows=5)
    #设置chunksize参数，来控制每次迭代数据的大小
    chunker = pd.read_csv("./train.csv",chunksize=5)
    for item in chunker:
        print(type(item))
        #<class 'pandas.core.frame.DataFrame'>
        print(len(item))
        #5
# 查看数据集的样本个数和原始特征维度：
    data_test_a.shape
    data_train.shape
    data_train.columns
    data_train.info()#通过info()来熟悉数据类型
    data_train.describe()#总体粗略的查看数据集各个特征的一些基本统计量
    data_train.head(3).append(data_train.tail(3))
# 查看数据集中特征缺失值，唯一值等：
    print(f'There are {data_train.isnull().any().sum()} columns in train dataset with missing values.')
# 查看特征的数值类型，对象类型：
    numerical_fea = list(data_train.select_dtypes(exclude=['object']).columns)
    category_fea = list(filter(lambda x: x not in numerical_fea,list(data_train.columns)))
    numerical_fea
    category_fea
# 用pandas_profiling生成数据报告：
    import pandas_profiling
    pfr = pandas_profiling.ProfileReport(data_train)
    pfr.to_file("./example.html")
总结：
数据探索性分析是我们初步了解数据，熟悉数据为特征工程做准备的阶段，甚至很多时候EDA阶段提取出来的特征可以直接当作规则来用。可见EDA的重要性，这个阶段的主要工作还是借助于各个简单的统计量来对数据整体的了解，分析各个类型变量相互之间的关系，以及用合适的图形可视化出来直观观察。


