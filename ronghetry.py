#coding:utf-8
import pandas as pd
import lightgbmClassifier as lc
import lightgbm  as lgb
import xgboostClassifier as xc
import xgboost as xgb
from sklearn.metrics import roc_auc_score
from sklearn.cross_validation import KFold
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
# 假设抽取的样本同分布 则 test sets 中含有 19493 正样本
import numpy as np

# 抽样选择数据
# 提取数据中的数值
def get_number(x):
    x = str(x)
    if '万元' in x:
        return float(x[0:-2])
    else:
        return float(x)

def normalize1(x, maxValue):
    return x / maxValue
# 根目录
dir = './public/'

print('reading train datas')
train = pd.read_csv(dir + 'train.csv')
# 训练集中标签为1的EID
pos_train = list(train[train['TARGET']==1]['EID'].unique())
# print(train.head())
print('train shape',train.shape)
org_train_shape = train.shape
print('positive sample',train[train.TARGET == 1].__len__())
print('positive ration',train[train.TARGET == 1].__len__() * 1.0/ len(train))
print('reading test datas')
test = pd.read_csv(dir + 'evaluation_public.csv')

# 全部的企业EID
all_eid_number = len(set(list(test['EID'].unique()) + list(train['EID'].unique())))
print('all EID number ',all_eid_number)

entbase = pd.read_csv(dir + '1entbase.csv')
alter = pd.read_csv(dir + '2alter.csv')
branch = pd.read_csv(dir + '3branch.csv')
invest = pd.read_csv(dir + '4invest.csv')
right = pd.read_csv(dir + '5right.csv')
project = pd.read_csv(dir + '6project.csv')
lawsuit = pd.read_csv(dir + '7lawsuit.csv')
breakfaith = pd.read_csv(dir + '8breakfaith.csv')
recruit = pd.read_csv(dir + '9recruit.csv')


# 获取企业基本信息表
# 获取企业基本信息表
entbase['ZCZB']=np.log1p(entbase['ZCZB'])
ZCZB=entbase['ZCZB']
maxValue=ZCZB.max()
entbase['ZCZB']=ZCZB.fillna(int(ZCZB.mean()))
entbase['ZCZB']=entbase['ZCZB'].map(lambda x:normalize1(x,maxValue))
# 题目要求是用0填充，因此对nan进行填充
entbase = entbase.fillna(0)


#根据变更类型进行独热编码
entbase=entbase[entbase['ZCZB']<100000]
entbase_ETYPE = pd.get_dummies(entbase['ETYPE'],prefix='ETYPE')
entbase_ETYPE_merge = pd.concat([entbase['EID'],entbase_ETYPE],axis=1)
# print(entbase_ETYPE_merge.head(15))
del entbase_ETYPE_merge['ETYPE_2.0']
# del entbase_ETYPE_merge['ETYPE_1.0']
# del entbase_ETYPE_merge['ETYPE_4.0']
del entbase['ETYPE']
# del entbase['TZINUM']
entbase= pd.merge(entbase,entbase_ETYPE_merge,on=['EID'])


#提取变更特征
print('aleter shape',alter.shape)
print('alter in EID number ratio',len(alter['EID'].unique())*1.0 / all_eid_number)


alter = alter.fillna(0)

# print('ALTERNO to cateary')
ALTERNO_to_index = list(alter['ALTERNO'].unique())
# 1 2 有金钱变化
alter['ALTERNO'] = alter['ALTERNO'].map(ALTERNO_to_index.index)

alter['ALTAF'] = np.log1p(alter['ALTAF'].map(get_number))
ALTAF=alter['ALTAF']
maxA=ALTAF.max()
alter['ALTAF']=alter['ALTAF'].fillna(int(ALTAF.min()))
alter['ALTAF']=alter['ALTAF'].map(lambda x:normalize1(x,maxA))

alter['ALTBE'] = np.log1p(alter['ALTBE'].map(get_number))
ALTBE=alter['ALTBE']
maxA=ALTBE.max()
alter['ALTBE']=alter['ALTBE'].fillna(int(ALTBE.min()))
alter['ALTBE']=alter['ALTBE'].map(lambda x:normalize1(x,maxA))
alter['ALTAF_ALTBE'] = alter['ALTAF'] - alter['ALTBE']

alter['ALTDATE_YEAR'] = alter['ALTDATE'].map(lambda x:x.split('-')[0]).astype(int)
alter['ALTDATE_MONTH'] = alter['ALTDATE'].map(lambda x:x.split('-')[1]).astype(int)
alter = alter.sort_values(['ALTDATE_YEAR','ALTDATE_MONTH'],ascending=True)

# 标签化 ALTERNO
#根据变更类型进行独热编码
# print(alter['ALTERNO'])
alter_ALTERNO = pd.get_dummies(alter['ALTERNO'],prefix='ALTERNO')
alter_ALTERNO_merge = pd.concat([alter['EID'],alter_ALTERNO],axis=1)


alter_ALTERNO_info_sum = alter_ALTERNO_merge.groupby(['EID'],as_index=False).sum()
alter_ALTERNO_info_count = alter_ALTERNO_merge.groupby(['EID'],as_index=False).count()  #记录变更次数
alter_ALTERNO_info_ration = alter_ALTERNO_merge.groupby(['EID']).sum() / alter_ALTERNO_merge.groupby(['EID']).count()
# print(alter_ALTERNO_info_ration.head(15))
alter_ALTERNO_info_ration = alter_ALTERNO_info_ration.reset_index()
# print(alter_ALTERNO_info_ration.head(15))

# 变更的第一年
alter_first_year = pd.DataFrame(alter[['EID','ALTDATE_YEAR']]).drop_duplicates(['EID'])
alter_first_month=pd.DataFrame(alter[['EID','ALTDATE_MONTH']]).drop_duplicates(['EID'])
alter_first_year.rename(columns={'ALTDATE_YEAR':'first_year'},inplace=True)
alter_first_month.rename(columns={'ALTDATE_MONTH':'first_month'},inplace=True)
# 变更的最后一年
alter = alter.sort_values(['ALTDATE_YEAR','ALTDATE_MONTH'],ascending=False)
alter_last_year = pd.DataFrame(alter[['EID','ALTDATE_YEAR']]).drop_duplicates(['EID'])
alter_last_month=pd.DataFrame(alter[['EID','ALTDATE_MONTH']]).drop_duplicates(['EID'])
alter_last_year.rename(columns={'ALTDATE_YEAR':'last_year'},inplace=True)
alter_last_month.rename(columns={'ALTDATE_MONTH':'last_month'},inplace=True)

alter_ALTERNO_info = pd.merge(alter_ALTERNO_info_sum,alter[['ALTAF_ALTBE','EID']],on=['EID']).drop_duplicates(['EID'])
alter_ALTERNO_info = pd.merge(alter_ALTERNO_info,alter_last_year,on=['EID'])
alter_ALTERNO_info=pd.merge(alter_ALTERNO_info,alter_last_month,on=['EID'],how='left')
alter_ALTERNO_info = alter_ALTERNO_info.fillna(-1)
# del alter_ALTERNO_info['ALTERNO_11']
# del alter_ALTERNO_info['ALTERNO_10']


# print branch
branch['B_ENDYEAR'] = branch['B_ENDYEAR'].fillna(branch['B_REYEAR'])
# print(branch['B_ENDYEAR'])
branch['sub_life'] = branch['B_ENDYEAR'].fillna(branch['B_REYEAR']) - branch['B_REYEAR']
# 筛选数据
branch = branch[branch['sub_life']>=0]
branch_count = branch.groupby(['EID'],as_index=False)['TYPECODE'].count()
branch_count.rename(columns = {'TYPECODE':'branch_count'},inplace=True)
branch = pd.merge(branch,branch_count,on=['EID'],how='left')
branch['branch_count'] = np.log1p(branch['branch_count'])
branch['branch_count'] = branch['branch_count'].astype(int)
branch['survive_num']=branch.groupby(['EID'])['sub_life'].count()
branch['sub_life'] = branch['sub_life'].replace({0.0:-1})
# print(branch)
home_prob = branch.groupby(by=['EID'])['IFHOME'].sum()/ branch.groupby(by=['EID'])['IFHOME'].count()
home_prob = home_prob.reset_index()
# bran_last_year=pd.DataFrame(branch[['EID','B_REYEAR']]).sort_values(['B_REYEAR'],ascending=False).drop_duplicates(['EID'])
branch_year=branch.groupby(['EID'])['B_REYEAR'].mean()
branch_year=branch_year.reset_index()
print(branch_year)
branch = pd.DataFrame(branch[['EID','sub_life']]).drop_duplicates('EID')
branch = pd.merge(branch,home_prob,on=['EID'],how='left')
branch = pd.merge(branch,branch_year,on=['EID'])
# print(branch)


invest['BTENDYEAR'] = invest['BTENDYEAR'].fillna(invest['BTYEAR'])
invest['invest_life'] = invest['BTENDYEAR'] - invest['BTYEAR']

invest_BTBL_sum = invest.groupby(['EID'],as_index=False)['BTBL'].sum()
invest_BTBL_sum.rename(columns={'BTBL':'BTBL_SUM'},inplace=True)

invest_BTBL_count = invest.groupby(['EID'],as_index=False)['BTBL'].count()
invest_BTBL_count.rename(columns={'BTBL':'BTBL_COUNT'},inplace=True)

BTBL_INFO = pd.merge(invest_BTBL_sum,invest_BTBL_count,on=['EID'],how='left')
BTBL_INFO['BTBL_RATIO'] = BTBL_INFO['BTBL_SUM'] / BTBL_INFO['BTBL_COUNT']
invest_home_prob=invest.groupby(by=['EID'])['IFHOME'].sum()/invest.groupby(by=['EID'])['IFHOME'].count()
invest_home_prob=invest_home_prob.reset_index()
invest_home_prob.rename(columns={'IFHOME':'invest_home_prob'},inplace=True)
print(invest_home_prob)
invest['invest_life'] = invest['invest_life'] > 0
invest['invest_life'] = invest['invest_life'].astype(int)
invest_life_ratio = invest.groupby(['EID'])['invest_life'].sum() / invest.groupby(['EID'])['invest_life'].count()
invest_life_ratio = invest_life_ratio.reset_index()
invest_life_ratio.rename(columns={'invest_life':'invest_life_ratio'},inplace=True)
invest_last_year = invest.sort_values('BTYEAR',ascending=False).drop_duplicates('EID')[['EID','BTYEAR']]
invest_first_year = invest.sort_values('BTYEAR').drop_duplicates('EID')[['EID','BTYEAR']]
invest_years=invest.groupby(['EID'])['BTYEAR'].mean()
invest_years=invest_years.reset_index()

invest = pd.merge(invest[['EID']],BTBL_INFO,on=['EID'],how='left').drop_duplicates(['EID'])
invest =pd.merge(invest[['EID']],invest_home_prob,on=['EID'],how='left')
# invest=pd.merge(invest,invest_years,on=['EID'],how='left')
invest = pd.merge(invest,invest_life_ratio,on=['EID'],how='left')
invest = pd.merge(invest,invest_last_year,on=['EID'],how='left')

# print(invest)
print('right//////////////')
#权利独热码
right_RIGHTTYPE = pd.get_dummies(right['RIGHTTYPE'],prefix='RIGHTTYPE')
right_RIGHTTYPE_info = pd.concat([right['EID'],right_RIGHTTYPE],axis=1)

right_RIGHTTYPE_info_sum = right_RIGHTTYPE_info.groupby(['EID'],as_index=False).sum().drop_duplicates(['EID'])
# print(right_RIGHTTYPE_info_sum)
right['ASKDATE_Y'] = right['ASKDATE'].map(lambda x:x.split('-')[0]).astype(int)
right['ASKDATE_M']=right['ASKDATE'].map(lambda x:x.split('-')[1]).astype(int)
right_last_year = right.sort_values('ASKDATE_Y',ascending=False).drop_duplicates('EID')[['EID','ASKDATE_Y']]
right_last_year.rename(columns={'ASKDATE_Y':'right_last_year'},inplace=True)
right_last_month=pd.DataFrame(right[['EID','ASKDATE_M']]).sort_values('ASKDATE_M',ascending=False).drop_duplicates('EID')
right_last_month.rename(columns={'ASKDATE_M':'right_last_month'},inplace=True)
right=right.fillna(0)
right_temp=right[right['FBDATE']!=0]
'''
right['right_apply']=right.groupby(by=['EID'])['FBDATE'].count()
right['right_get']=right.groupby(by=['EID'])['FBDATE'].count()
right['right_get_prob']=right['right_get']/right['right_apply']
'''
right_get_rate=right_temp.groupby(by=['EID'])['FBDATE'].count()/right.groupby(by=['EID'])['FBDATE'].count()
right_get_rate=right_get_rate.reset_index()
right_get_rate.rename(columns={'FBDATE':'right_get_rate'},inplace=True)
right_get_rate=right_get_rate.fillna(0)
right=right.fillna(0)
right_count = right.groupby(['EID'],as_index=False)['RIGHTTYPE'].count()

right_count.rename(columns={'RIGHTTYPE':'right_count'},inplace=True)
right = pd.merge(right[['EID']],right_RIGHTTYPE_info_sum,on=['EID'],how='left').drop_duplicates(['EID'])
right = pd.merge(right,right_last_year,on=['EID'],how='left')
right= pd.merge(right,right_last_month,on=['EID'],how='left')
right = pd.merge(right,right_count,on=['EID'],how='left')
right=pd.merge(right,right_get_rate,on=['EID'],how='left')

print(right)


project['DJDATE_Y'] = project['DJDATE'].map(lambda x:x.split('-')[0])
project['DJDATE_M'] = project['DJDATE'].map(lambda x:x.split('-')[1])
# project_DJDATE_Y = pd.get_dummies(project['DJDATE_Y'],prefix='DJDATE')
# project_DJDATE_Y_info = pd.concat([project['EID'],project_DJDATE_Y],axis=1)
# project_DJDATE_Y_info_sum = project_DJDATE_Y_info.groupby(['EID'],as_index=False).sum()
# project_DJDATE_Y_info_sum = project_DJDATE_Y_info_sum.drop_duplicates(['EID'])

project_last_year = project.sort_values('DJDATE_Y',ascending=False).drop_duplicates('EID')[['EID','DJDATE_Y']]
project_last_month=pd.DataFrame(project[['EID','DJDATE_M']]).sort_values('DJDATE_M',ascending=False).drop_duplicates('EID')[['EID','DJDATE_M']]

project_home_prob = project.groupby(by=['EID'])['IFHOME'].sum()/ project.groupby(by=['EID'])['IFHOME'].count()
project_home_prob = project_home_prob.reset_index()
project_home_prob.rename(columns={'IFHOME':'project_home_prob'},inplace=True)

project = pd.merge(project[['EID']],project_last_year,on=['EID'],how='left').drop_duplicates(['EID'])
project=pd.merge(project[['EID']],project_last_month,on=['EID'],how='left')
# project = pd.merge(project,project_DJDATE_Y_info_sum,on=['EID'],how='left')
project=pd.merge(project,project_home_prob,on=['EID'],how='left')

print(project)
money_sum=lawsuit[['EID','LAWAMOUNT']].groupby('EID').sum()
money_sum=money_sum.rename(columns={"LAWAMOUNT":"money_sum"})
zczb=entbase[['EID','ZCZB']].copy()
zczb=zczb[zczb['EID'].isin(money_sum.index)]
zczb=zczb.rename(columns={"ZCZB":"money_sum"})
zczb=zczb.groupby('EID').max()
money_BL=money_sum.div(zczb)
money_BL=money_BL.rename(columns={"money_sum":"money_BL"})
money_BL['money_BL']=money_BL['money_BL']/10000
money_BL=money_BL.reset_index()

lawsuit_LAWAMOUNT_sum = lawsuit.groupby(['EID'],as_index=False)['LAWAMOUNT'].sum()
lawsuit_LAWAMOUNT_sum.rename(columns={'LAWAMOUNT':'lawsuit_LAWAMOUNT_sum'},inplace=True)
lawsuit_LAWAMOUNT_sum['lawsuit_LAWAMOUNT_sum'] = np.log1p(lawsuit_LAWAMOUNT_sum['lawsuit_LAWAMOUNT_sum'])
lawsuit_LAWAMOUNT_sum['lawsuit_LAWAMOUNT_sum'] = lawsuit_LAWAMOUNT_sum['lawsuit_LAWAMOUNT_sum'].astype(int)
lawsuit_LAWAMOUNT_count = lawsuit.groupby(['EID'],as_index=False)['LAWAMOUNT'].count()

lawsuit_LAWAMOUNT_count.rename(columns={'LAWAMOUNT':'lawsuit_LAWAMOUNT_count'},inplace=True)
lawsuit_LAWAMOUNT_avg=lawsuit.groupby(['EID'],as_index=False)['LAWAMOUNT'].sum()/lawsuit.groupby(by=['EID'],as_index=False)['LAWAMOUNT'].count()

lawsuit['LAWDATE_Y'] = lawsuit['LAWDATE'].map(lambda x:x.split('-')[0])
lawsuit['LAWDATE_M'] = lawsuit['LAWDATE'].map(lambda x:x.split('-')[1])
lawsuit_last_year = lawsuit.sort_values('LAWDATE_Y',ascending=False).drop_duplicates('EID')[['EID','LAWDATE_Y']]
lawsuit_last_month=lawsuit.sort_values('LAWDATE_M',ascending=False).drop_duplicates('EID')[['EID','LAWDATE_M']]
lawsuits=pd.merge(lawsuit_last_year,lawsuit_LAWAMOUNT_avg,on=['EID'],how='left')
lawsuits=pd.merge(lawsuit_last_year,lawsuit_last_month,on=['EID'],how='left')
lawsuits=pd.merge(lawsuits,lawsuit_LAWAMOUNT_count,on=['EID'],how='left')
lawsuits=pd.merge(lawsuits,lawsuit_LAWAMOUNT_sum,on=['EID'],how='left')


breakfaith['FBDATE_Y'] = breakfaith['FBDATE'].map(lambda x:x.split('/')[0])
breakfaith_first_year = breakfaith.sort_values('FBDATE_Y').drop_duplicates('EID')[['EID','FBDATE_Y']]

breakfaith['SXENDDATE'] = breakfaith['SXENDDATE'].fillna(0)
breakfaith['is_breakfaith'] = breakfaith['SXENDDATE']!=0
breakfaith['is_breakfaith'] = breakfaith['is_breakfaith'].astype(int)

breakfaith_is_count = breakfaith.groupby(['EID'],as_index=False)['is_breakfaith'].count()
breakfaith_is_sum = breakfaith.groupby(['EID'],as_index=False)['is_breakfaith'].sum()

breakfaith_is_count.rename(columns={'is_breakfaith':'breakfaith_is_count'},inplace=True)
breakfaith_is_sum.rename(columns={'is_breakfaith':'breakfaith_is_sum'},inplace=True)
breakfaith_is_info=breakfaith_is_count
#breakfaith_is_info = pd.merge(breakfaith_is_count,breakfaith_is_sum,on=['EID'],how='left')
#breakfaith_is_info['ratio'] = breakfaith_is_info['breakfaith_is_sum'] / breakfaith_is_info['breakfaith_is_count']
#del breakfaith_is_info['breakfaith_is_sum']


recruit['RECDATE_Y'] = recruit['RECDATE'].map(lambda x:x.split('-')[0])
recruit_train_last_year = recruit.sort_values('RECDATE_Y',ascending=False).drop_duplicates('EID')[['EID','RECDATE_Y']]
recruit_WZCODE = pd.get_dummies(recruit['WZCODE'],prefix='WZCODE')
recruit_WZCODE_merge = pd.concat([recruit['EID'],recruit_WZCODE],axis=1)
# 1
recruit_WZCODE_info_sum = recruit_WZCODE_merge.groupby(['EID'],as_index=False).sum().drop_duplicates(['EID'])
# 2
recruit['RECRNUM'] = recruit['RECRNUM'].fillna(0)
recruit_RECRNUM_count = recruit.groupby(['EID'],as_index=False)['RECRNUM'].count()
recruit_RECRNUM_count.rename(columns={'RECRNUM':'recruit_RECRNUM_count'},inplace=True)
# 3
recruit_RECRNUM_sum = recruit.groupby(['EID'],as_index=False)['RECRNUM'].sum()
recruit_RECRNUM_sum.rename(columns={'RECRNUM':'recruit_RECRNUM_sum'},inplace=True)
recruit_RECRNUM_sum['recruit_RECRNUM_sum'] = recruit_RECRNUM_sum['recruit_RECRNUM_sum']
# 4
recruit_RECRNUM_info = pd.merge(recruit[['EID']],recruit_RECRNUM_sum,on=['EID']).drop_duplicates(['EID'])
recruit_RECRNUM_info = pd.merge(recruit_RECRNUM_info,recruit_RECRNUM_count,on=['EID'])
recruit_RECRNUM_info['recurt_info_ration'] = recruit_RECRNUM_info['recruit_RECRNUM_sum'] / recruit_RECRNUM_info['recruit_RECRNUM_count']


print('merge train/test')
train = pd.merge(train,entbase,on=['EID'],how='left')
# 根据注册资本简单筛选样本
print('select sample to train set...')

train = pd.merge(train,alter_ALTERNO_info,on=['EID'],how='left')
train = pd.merge(train,branch,on=['EID'],how='left')
train = pd.merge(train,right,on=['EID'],how='left')
train = pd.merge(train,invest,on=['EID'],how='left')
train = pd.merge(train,project,on=['EID'],how='left')
train = pd.merge(train,lawsuit_LAWAMOUNT_count,on=['EID'],how='left')
train=pd.merge(train,lawsuit_last_year,on=['EID'],how='left')
train = pd.merge(train,breakfaith_is_info,on=['EID'],how='left')
train = pd.merge(train,recruit_WZCODE_info_sum,on=['EID'],how='left')
train = pd.merge(train,recruit_RECRNUM_info,on=['EID'],how='left')
train=pd.merge(train,money_BL,on=['EID'],how='left')

test = pd.merge(test,entbase,on=['EID'],how='left')
test = pd.merge(test,alter_ALTERNO_info,on=['EID'],how='left')
test = pd.merge(test,branch,on=['EID'],how='left')
test = pd.merge(test,right,on=['EID'],how='left')
test = pd.merge(test,invest,on=['EID'],how='left')
test = pd.merge(test,project,on=['EID'],how='left')
test = pd.merge(test,lawsuit_LAWAMOUNT_count,on=['EID'],how='left')
test = pd.merge(test,lawsuit_last_year,on=['EID'],how='left')
test = pd.merge(test,breakfaith_is_info,on=['EID'],how='left')
test = pd.merge(test,recruit_WZCODE_info_sum,on=['EID'],how='left')
test = pd.merge(test,recruit_RECRNUM_info,on=['EID'],how='left')
test=pd.merge(test,money_BL,on=['EID'],how='left')
test = test.fillna(-999)
train = train.fillna(-999)

# train=shuffle(train)
del train['EID']
test_index = test.pop('EID')
y_label = train.pop('TARGET')

train=train.values
y_label=y_label.values
test=test.values
#
# tmp1 = train[train.TARGET==1]
# tmp0 = train[train.TARGET==0]
# x_valid_1 = tmp1.sample(frac=0.3, random_state=70, axis=0)
# x_train_1 = tmp1.drop(x_valid_1.index.tolist())
# x_valid_2 = tmp0.sample(frac=0.1, random_state=70, axis=0)
# x_train_2 = tmp0.drop(x_valid_2.index.tolist())
# X_train = pd.concat([x_train_1,x_train_2],axis=0)
# y_train = X_train.pop('TARGET')
# X_test = pd.concat([x_valid_1,x_valid_2],axis=0)
# y_test = X_test.pop('TARGET')
# feature_len = X_train.shape[1]
# feature=X_train.columns
# X_train = X_train.values
# X_test = X_test.values
# y_train = y_train.values
# y_test = y_test.values

l_params={
    'task': 'train',
    'boosting_type': 'gbdt',
    'objective': 'binary',
    'metric': {'auc'},
    'num_leaves': 128,
    'learning_rate': 0.05,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'verbose': 0,
}
x_params={
    'booster':'gbtree',
    'objective':'binary:logistic',
    'eta':0.05,
    'max_depth':6,
    'subsample': 0.6,
    'min_child_weight': 12,
    'colsample_bytree': 0.8,
    'scale_pos_weight': 1,
    'eval_metric': 'auc',
    'gamma': 0.2,
    'silent':1,
    'lambda': 200,
}
clfs = [
    lc.LightGbmClassifier(params=l_params,num_boost_round=1000,early_stopping=50),
    xc.XGBoostClassifier(params=x_params, num_boost_round=800, early_stopping=50)
]

dataset_blend_train = np.zeros((train.shape[0], len(clfs)))
dataset_blend_test = np.zeros((test.shape[0], len(clfs)))

n_folds=8
skf=list(KFold(len(y_label),n_folds=n_folds, shuffle=True, random_state=2017))

for j,clf in enumerate(clfs):
    dataset_blend_test_j=np.zeros((test.shape[0],len(skf)))
    for i,(tr,te) in enumerate(skf):

        X_train,y_train,X_valid,y_valid=train[tr],y_label[tr],train[te],y_label[te]
        clf.fit(X_train,y_train,X_valid,y_valid)
        y_submission=clf.predict(X_valid)
        dataset_blend_train[te,j]=y_submission
        dataset_blend_test_j[:,i]=clf.predict(test)
    dataset_blend_test[:,j]=dataset_blend_test_j.mean(1)

print(dataset_blend_train.shape)
X_train,X_test,y_train,y_test=train_test_split(dataset_blend_train,y_label,test_size=0.3,random_state=10000)
data_train=lgb.Dataset(X_train,label=y_train)
data_valid=lgb.Dataset(X_test,label=y_test,reference=data_train)
model=lgb.train(l_params,
                data_train,
                num_boost_round=1000,
                valid_sets=data_valid,
                early_stopping_rounds=50
                )
y_pred=model.predict(dataset_blend_test,num_iteration=model.best_iteration)
# data_train=xgb.DMatrix(X_train,label=y_train)
# data_eval=xgb.DMatrix(X_test,label=y_test)
# watch_list=[(data_eval,'valid'),(data_train,'train')]
# xgb_test=xgb.DMatrix(dataset_blend_test)
# model=xgb.train(x_params,
#                 data_train,
#                 evals=watch_list,
#                 num_boost_round = 500,
#                 early_stopping_rounds=50,
#              )
# y_pred=model.predict(xgb_test)
y_pred = np.round(y_pred,8)
result = pd.DataFrame({'PROB':list(y_pred),})
result['FORTARGET'] = result['PROB'] > 0.34
result['PROB'] = result['PROB'].astype('str')
result['FORTARGET'] = result['FORTARGET'].astype('int')
result = pd.concat([test_index,result],axis=1)
print('positive sample',result[result.FORTARGET == 1].__len__())
print('positive ration',result[result.FORTARGET == 1].__len__() * 1.0/ len(result))

result = pd.DataFrame(result).drop_duplicates(['EID'])
print(result)
result[['EID','FORTARGET','PROB']].to_csv('./evaluation_public.csv',index=None)
