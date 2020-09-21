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
1.若这个变量是有大小关系的，比如该数据中的grade，可以使用数值映射
（'A':1,'B':2,'C':3,'D':4,'E':5,'F':6,'G':7）
2.若变量之间是没有什么关系的，我们可以用独热编码（One-hot-encode）

	# 部分类别特征
	cate_features = ['grade', 'subGrade', 'employmentTitle', 'homeOwnership', 'verificationStatus', 'purpose', 'postCode', 'regionCode', \
			 'applicationType', 'initialListStatus', 'title', 'policyCode']
	for f in cate_features:
	    print(f, '类型数：', data[f].nunique())

	#对于grade这种有大小，优劣区分的特征
	for data in [data_train, data_test_a]:
	    data['grade'] = data['grade'].map({'A':1,'B':2,'C':3,'D':4,'E':5,'F':6,'G':7})

	#对于这些没有大小之分的特征,类型数在2之上，又不是高维稀疏的,且纯分类特征,可以使用独热编码
	for data in [train_data,test_A_data]:
	    data = pd.get_dummies(data,columns=['subGrade', 'homeOwnership', 'verificationStatus','regionCode'],drop_first= True)
# 2 异常值处理
当发现异常值后，一定要先分清是什么原因导致的异常值，然后再考虑如何处理。首先，如果这一异常值并不代表一种规律性的，而是极其偶然的现象，或者说并不想研究这种偶然的现象，这时可以将其删除。其次，如果异常值存在且代表了一种真实存在的现象，那就不能随便删除。在现有的欺诈场景中很多时候欺诈数据本身相对于正常数据勒说就是异常的，要把这些异常点纳入，重新拟合模型，研究其规律。能用监督的用监督模型，不能用的还可以考虑用异常检测的算法来做。

注意test的数据不能删。

	检测异常的方法一：均方差

	在统计学中，如果一个数据分布近似正态，那么大约 68% 的数据值会在均值的一个标准差范围内，大约 95% 会在两个标准差范围内，大约 99.7% 会在三个标准差范围内。

	def find_outliers_by_3segama(data,fea):
	    data_std = np.std(data[fea])
	    data_mean = np.mean(data[fea])
	    outliers_cut_off = data_std * 3
	    lower_rule = data_mean - outliers_cut_off
	    upper_rule = data_mean + outliers_cut_off
	    data[fea+'_outliers'] = data[fea].apply(lambda x:str('异常值') if x > upper_rule or x < lower_rule else '正常值')
	    return data
	    
	检测异常的方法二：箱型图
	四分位数会将数据分为三个点和四个区间，IQR = Q3 -Q1，下触须=Q1 − 1.5x IQR，上触须=Q3 + 1.5x IQR；
	
# 3 数据分桶
特征分箱的目的：
1.从模型效果上来看，特征分箱主要是为了降低变量的复杂性，减少变量噪音对模型的影响，提高自变量和因变量的相关度。从而使模型更加稳定。
数据分桶的对象：
1.将连续变量离散化
2.将多状态的离散变量并成少状态
分箱的原因：
1.数据的特征内的值跨度可能比较大，对有监督和无监督中如k-均值聚类它使用欧氏距离作为相似度函数来测量数据点之间的相似度。都会造成大吃小的影响，其中一种解决方法是对计数值进行区间量化即数据分桶也叫做数据分箱，然后使用量化后的结果。
分箱的优点：
1.处理缺失值上，可以将null单独作为一个分箱
2.处理异常值上，可以把其通过分箱离散化处理，从而提高变量的鲁棒性（抗干扰能力）。例如，age若出现200这种异常值，可分入“age > 60”这个分箱里，排除影响。
3.业务解释性上，我们习惯于线性判断变量的作用，当x越来越大，y就越来越大。但实际x与y之间经常存在着非线性关系，此时可经过WOE（Weight of Evidence）变换。
特别要注意一下分箱的基本原则：
1.最小分箱占比不低于5%
2.箱内不能全是好客户
3.连续箱单调
固定宽度分箱
当数值横跨多个数量级时，最好按照 10 的幂（或任何常数的幂）来进行分组。固定宽度分箱非常容易计算，但如果计数值中有比较大的缺口，就会产生很多没有任何数据的空箱子。

	# 通过除法映射到间隔均匀的分箱中，每个分箱的取值范围都是loanAmnt/1000
	data['loanAmnt_bin1'] = np.floor_divide(data['loanAmnt'],1000)

	# 通过对数函数映射到指数宽度分箱
	data['loanAmnt_bin1'] = np.floor(np.log10(data['loanAmnt'])

	# 分位数分箱
	data['loanAmnt_bin3'] = pd.qcut(data['loanAmnt'], 10, labels=False)

	#卡方分箱及其他
# 4 特征交互
交互特征的构造非常简单，使用起来却代价不菲。如果线性模型中包含有交互特征对，那它的训练时间和评分时间就会从 O(n) 增加到 O(n2)，其中 n 是单一特征的数量。

	for col in ['grade', 'subGrade']: 
	    temp_dict = train_data.groupby([col])['isDefault'].agg(['mean']).reset_index().rename(columns={'mean': col + '_target_mean'})
	    temp_dict.index = temp_dict[col].values
	    temp_dict = temp_dict[col + '_target_mean'].to_dict()

	    train_data[col + '_target_mean'] = train_data[col].map(temp_dict)
	    train_data[col + '_target_mean'] = test_A_data[col].map(temp_dict)


	# 其他衍生变量 mean 和 std
	for df in [train_data, test_A_data]:
	    for item in ['n0','n1','n2','n2.1','n4','n5','n6','n7','n8','n9','n10','n11','n12','n13','n14']:
		df['grade_to_mean_' + item] = df['grade'] / df.groupby([item])['grade'].transform('mean')
		df['grade_to_std_' + item] = df['grade'] / df.groupby([item])['grade'].transform('std')
# 5 特征编码
labelEncoder 直接放入模型中

	##label-encode:subGrade,postCode,title
	# 高维类别特征需要进行转换
	from sklearn.preprocessing import LabelEncoder
	#tqdm是一个看程序进程的函数
	for col in tqdm(['subGrade','postCode','title','employmentTitle']):
	    le = LabelEncoder()
	    # astype()函数可用于转化dateframe某一列的数据类型,values方法返回结果是数组
	    le.fit(list(train_data[col].astype(str).values) + list(test_A_data[col].astype(str).values))
	    print(le.classes_)
	    train_data[col] = le.transform(list(train_data[col].astype(str).values))
	    test_A_data[col] = le.transform(list(test_A_data[col].astype(str).values))
	print('Label Encoding Finished!')
	
逻辑回归等模型要单独增加的特征工程

对特征做归一化，去除相关性高的特征

归一化目的是让训练过程更好更快的收敛，避免特征大吃小的问题

去除相关性是增加模型的可解释性，加快预测过程。

	# 举例归一化过程
	#伪代码
	for fea in [要归一化的特征列表]：
	    data[fea] = ((data[fea] - np.min(data[fea])) / (np.max(data[fea]) - np.min(data[fea])))

# 7. 特征选择
特征选择技术可以精简掉无用的特征，以降低最终模型的复杂性，它的最终目的是得到一个简约模型，在不降低预测准确率或对预测准确率影响不大的情况下提高计算速度。特征选择不是为了减少训练时间（实际上，一些技术会增加总体训练时间），而是为了减少模型评分时间。
1.filter（过滤法）

	方差选择法
	相关系数法（pearson 相关系数）
	卡方检验
	互信息法
2.Wrapper（包裹法）

	RFE（递归特征消除）
	RFECV（递归特征消除交叉验证）
	
3.Embedded

	基于惩罚项的特征选择法
	基于树模型的特征选择
# 代码汇总

	# 删除不需要的数据
	for data in [data_train, data_test_a]:
	    data.drop(['issueDate','id'], axis=1,inplace=True)

	#纵向用缺失值上面的值替换缺失值
	data_train = data_train.fillna(axis=0,method='ffill')

	x_train = data_train.drop(['isDefault','id'], axis=1)
	#计算协方差
	data_corr = x_train.corrwith(data_train.isDefault) #计算相关性
	result = pd.DataFrame(columns=['features', 'corr'])
	result['features'] = data_corr.index
	result['corr'] = data_corr.values

	# 当然也可以直接看图
	data_numeric = data_train[numerical_fea]
	correlation = data_numeric.corr()

	f , ax = plt.subplots(figsize = (7, 7))
	plt.title('Correlation of Numeric Features with Price',y=1,size=16)
	sns.heatmap(correlation,square = True,  vmax=0.8)

	features = [f for f in data_train.columns if f not in ['id','issueDate','isDefault'] and '_outliers' not in f]
	x_train = data_train[features]
	x_test = data_test_a[features]
	y_train = data_train['isDefault']

	def cv_model(clf, train_x, train_y, test_x, clf_name):
	    folds = 5
	    seed = 2020
	    kf = KFold(n_splits=folds, shuffle=True, random_state=seed)

	    train = np.zeros(train_x.shape[0])
	    test = np.zeros(test_x.shape[0])

	    cv_scores = []

	    for i, (train_index, valid_index) in enumerate(kf.split(train_x, train_y)):
		print('************************************ {} ************************************'.format(str(i+1)))
		trn_x, trn_y, val_x, val_y = train_x.iloc[train_index], train_y[train_index], train_x.iloc[valid_index], train_y[valid_index]

		if clf_name == "lgb":
		    train_matrix = clf.Dataset(trn_x, label=trn_y)
		    valid_matrix = clf.Dataset(val_x, label=val_y)

		    params = {
			'boosting_type': 'gbdt',
			'objective': 'binary',
			'metric': 'auc',
			'min_child_weight': 5,
			'num_leaves': 2 ** 5,
			'lambda_l2': 10,
			'feature_fraction': 0.8,
			'bagging_fraction': 0.8,
			'bagging_freq': 4,
			'learning_rate': 0.1,
			'seed': 2020,
			'nthread': 28,
			'n_jobs':24,
			'silent': True,
			'verbose': -1,
		    }

		    model = clf.train(params, train_matrix, 50000, valid_sets=[train_matrix, valid_matrix], verbose_eval=200,early_stopping_rounds=200)
		    val_pred = model.predict(val_x, num_iteration=model.best_iteration)
		    test_pred = model.predict(test_x, num_iteration=model.best_iteration)

		    # print(list(sorted(zip(features, model.feature_importance("gain")), key=lambda x: x[1], reverse=True))[:20])

		if clf_name == "xgb":
		    train_matrix = clf.DMatrix(trn_x , label=trn_y)
		    valid_matrix = clf.DMatrix(val_x , label=val_y)

		    params = {'booster': 'gbtree',
			      'objective': 'binary:logistic',
			      'eval_metric': 'auc',
			      'gamma': 1,
			      'min_child_weight': 1.5,
			      'max_depth': 5,
			      'lambda': 10,
			      'subsample': 0.7,
			      'colsample_bytree': 0.7,
			      'colsample_bylevel': 0.7,
			      'eta': 0.04,
			      'tree_method': 'exact',
			      'seed': 2020,
			      'nthread': 36,
			      "silent": True,
			      }

		    watchlist = [(train_matrix, 'train'),(valid_matrix, 'eval')]

		    model = clf.train(params, train_matrix, num_boost_round=50000, evals=watchlist, verbose_eval=200, early_stopping_rounds=200)
		    val_pred  = model.predict(valid_matrix, ntree_limit=model.best_ntree_limit)
		    test_pred = model.predict(test_x , ntree_limit=model.best_ntree_limit)

		if clf_name == "cat":
		    params = {'learning_rate': 0.05, 'depth': 5, 'l2_leaf_reg': 10, 'bootstrap_type': 'Bernoulli',
			      'od_type': 'Iter', 'od_wait': 50, 'random_seed': 11, 'allow_writing_files': False}

		    model = clf(iterations=20000, **params)
		    model.fit(trn_x, trn_y, eval_set=(val_x, val_y),
			      cat_features=[], use_best_model=True, verbose=500)

		    val_pred  = model.predict(val_x)
		    test_pred = model.predict(test_x)

		train[valid_index] = val_pred
		test = test_pred / kf.n_splits
		cv_scores.append(roc_auc_score(val_y, val_pred))

		print(cv_scores)

	    print("%s_scotrainre_list:" % clf_name, cv_scores)
	    print("%s_score_mean:" % clf_name, np.mean(cv_scores))
	    print("%s_score_std:" % clf_name, np.std(cv_scores))
	    return train, test

	def lgb_model(x_train, y_train, x_test):
	    lgb_train, lgb_test = cv_model(lgb, x_train, y_train, x_test, "lgb")
	    return lgb_train, lgb_test

	def xgb_model(x_train, y_train, x_test):
	    xgb_train, xgb_test = cv_model(xgb, x_train, y_train, x_test, "xgb")
	    return xgb_train, xgb_test

	def cat_model(x_train, y_train, x_test):
	    cat_train, cat_test = cv_model(CatBoostRegressor, x_train, y_train, x_test, "cat")

	lgb_train, lgb_test = lgb_model(x_train, y_train, x_test)

	testA_result = pd.read_csv('../testA_result.csv')
	roc_auc_score(testA_result['isDefault'].values, lgb_test)































