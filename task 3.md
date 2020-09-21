# 学习目标
学习特征预处理、缺失值、异常值处理、数据分桶等特征处理方法

学习特征交互、编码、选择的相应方法
# 1.导入包并读取数据
	import pandas as pd
	import numpy as np
	import matplotlib.pyplot as plt
	import seaborn as sns
	import datetime
	from tqdm import tqdm
	from sklearn.preprocessing import LabelEncoder
	from sklearn.feature_selection import SelectKBest
	from sklearn.feature_selection import chi2
	from sklearn.preprocessing import MinMaxScaler
	import xgboost as xgb
	import lightgbm as lgb
	from catboost import CatBoostRegressor
	import warnings
	from sklearn.model_selection import StratifiedKFold, KFold
	from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, log_loss
	warnings.filterwarnings('ignore')

	#训练集测试集导入
	train_data = pd.read_csv('train.csv')
	test_A_data = pd.read_csv('testA.csv')
	#分离出数值型特征和类别型特征
	numerical_fea = list(train_data.select_dtypes(exclude = ['object']).columns)
	category_fea = list(train_data.select_dtypes(include = ['object']).columns)
	label = 'isDefault'
	numerical_fea.remove(label)
# 1.1特征预处理
缺失值填充

把所有缺失值替换为指定的值0

data_train = data_train.fillna(0)

向用缺失值上面的值替换缺失值

data_train = data_train.fillna(axis=0,method='ffill')

纵向用缺失值下面的值替换缺失值,且设置最多只填充两个连续的缺失值

data_train = data_train.fillna(axis=0,method='bfill',limit=2)

# 查看缺失值情况
	train_data.isnull().sum()

	'''
	results:
	id                        0
	loanAmnt                  0
	term                      0
	interestRate              0
	installment               0
	grade                     0
	subGrade                  0
	employmentTitle           1
	employmentLength      46799
	homeOwnership             0
	annualIncome              0
	verificationStatus        0
	issueDate                 0
	isDefault                 0
	purpose                   0
	postCode                  1
	regionCode                0
	dti                     239
	delinquency_2years        0
	ficoRangeLow              0
	ficoRangeHigh             0
	openAcc                   0
	pubRec                    0
	pubRecBankruptcies      405
	revolBal                  0
	revolUtil               531
	totalAcc                  0
	initialListStatus         0
	applicationType           0
	earliesCreditLine         0
	title                     1
	policyCode                0
	n0                    40270
	n1                    40270
	n2                    40270
	n2.1                  40270
	n4                    33239
	n5                    40270
	n6                    40270
	n7                    40270
	n8                    40271
	n9                    40270
	n10                   33239
	n11                   69752
	n12                   40270
	n13                   40270
	n14                   40270
	dtype: int64
	'''
看到n0-n14以及employLength特征缺失值较多，employmentTitle，postCode，dti，pubRecBankruptcies，revolUtil，title有较少的缺失，这里采用的方法是对于数值型变量，取中位数，对于类别型变量，使用众数来填充缺失值：
	train_data[numerical_fea] = train_data[numerical_fea].fillna(train_data[numerical_fea].median())
	train_data[category_fea] = train_data[category_fea].fillna(train_data[category_fea].mode())

	#重新查看一下缺失值的情况
	train_data.isnull().sum()
# 1.2时间格式处理
isissueDate（贷款发放的月份）这个特征是一个时间特征，处理的方式是计算借款日与最文件中最小的日期的距离的天数来构造一个新的特征：
	#最早的日期
	startdate = datetime.datetime.strptime('2007-06-01','%Y-%m-%d')
	#先转换格式再用日期-最早的日期得出天数为新的特征issueDateDT
	for data in [train_data,test_A_data]:
		data['issueDate'] =pd.to_datetime(data['issueDate'],format = '%Y-%m-%d')
		#构造时间特征
		data['issueDateDT'] = data['issueDate'].apply(lambda x: x - startdate).dt.days
接下来将employmentLength的数据做一些预处理:空值依然返回空值，>10年的全部分到10这个类别，<1年的分到0这个类别。
	def employmentLength_to_int(s):
		if pd.isnull(s):
			return s
		else:
			return np.int8(s.split(' ')[0])

	for data in [train_data,test_A_data]:
		data['employmentLength'].replace(to_replace='< 1 year',value='0 year',inplace=True)
		data['employmentLength'].replace(to_replace='10+ years',value='10 years',inplace=True)
		data['employmentLength'] = data['employmentLength'].apply(employmentLength_to_int)
对earliesCreditLine(借款人最早报告的信用额度开立的月份)进行预处理:我们直接提取出年份
	for data in [train_data,test_A_data]:
		data['earliesCreditLine'] = data['earliesCreditLine'].apply(lambda x : x[-4:])
# 1.3 类别型特征处理




















