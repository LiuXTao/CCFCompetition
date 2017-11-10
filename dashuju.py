#coding:utf-8
import pandas as pd
from sklearn.metrics import f1_score,auc
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
# 假设抽取的样本同分布 则 test sets 中含有 19493 正样本
import numpy as np
import lightgbm as lgb
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
entbase=entbase[entbase['ZCZB']<90000]

entbase['ZCZB']=np.log1p(entbase['ZCZB'])
ZCZB=entbase[['EID','ZCZB']]
maxValue=ZCZB['ZCZB'].max()
entbase['ZCZB']=entbase['ZCZB'].fillna(int(ZCZB['ZCZB'].mean()))
entbase['ZCZB']=entbase['ZCZB'].map(lambda x:normalize1(x,maxValue))

#根据变更类型进行独热编码
entbase_ETYPE = pd.get_dummies(entbase['ETYPE'],prefix='ETYPE')
entbase_ETYPE_merge = pd.concat([entbase['EID'],entbase_ETYPE],axis=1)
del entbase_ETYPE_merge['ETYPE_2.0']
# del entbase_ETYPE_merge['ETYPE_1.0']
# del entbase_ETYPE_merge['ETYPE_4.0']
del entbase['ETYPE']
entbase= pd.merge(entbase,entbase_ETYPE_merge,on=['EID'])
# del entbase['TZINUM']
# 题目要求是用0填充，因此对nan进行填充
entbase = entbase.fillna(0)
entbase['ZCZB_FINZB']=entbase['ZCZB']-entbase['FINZB']
#提取变更特征
print('aleter shape',alter.shape)
print('alter in EID number ratio',len(alter['EID'].unique())*1.0 / all_eid_number)


alter = alter.fillna(0)

# print('ALTERNO to cateary')
ALTERNO_to_index = list(alter['ALTERNO'].unique())
# 1 2 有金钱变化
alter['ALTERNO'] = alter['ALTERNO'].map(ALTERNO_to_index.index)

alter['ALTAF']=alter['ALTAF'].map(get_number)
alter=alter[alter['ALTAF']<2000000]
print(alter['ALTAF'].max())
alter['ALTAF'] = np.log1p(alter['ALTAF'])
ALTAF=alter['ALTAF']
maxA=ALTAF.max()
alter['ALTAF']=alter['ALTAF'].fillna(int(ALTAF.mean()))
alter['ALTAF']=alter['ALTAF'].map(lambda x:normalize1(x,maxA))

alter['ALTBE'] = np.log1p(alter['ALTBE'].map(get_number))
ALTBE=alter['ALTBE']
maxA=ALTBE.max()
alter['ALTBE']=alter['ALTBE'].fillna(int(ALTBE.mean()))
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
alter_ALTERNO_info_ration = alter_ALTERNO_info_ration.reset_index()


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

alter_year=alter.groupby(by=['EID'])['ALTDATE_YEAR'].mean().astype(int)
alter_year=alter_year.reset_index()

alter_month=alter.groupby(by=['EID'])['ALTDATE_MONTH'].mean().astype(int)
alter_month=alter_month.reset_index()

alter_ALTERNO_info = pd.merge(alter_ALTERNO_info_sum,alter[['ALTAF_ALTBE','EID']],on=['EID']).drop_duplicates(['EID'])
# alter_ALTERNO_info = pd.merge(alter_ALTERNO_info,alter_year,on=['EID'])
# alter_ALTERNO_info=pd.merge(alter_ALTERNO_info,alter_month,on=['EID'],how='left')
alter_ALTERNO_info=pd.merge(alter_ALTERNO_info,alter_last_year,on=['EID'],how='left')
alter_ALTERNO_info=pd.merge(alter_ALTERNO_info,alter_last_month,on=['EID'],how='left')
alter_ALTERNO_info = alter_ALTERNO_info.fillna(-1)
# del alter_ALTERNO_info['ALTERNO_11']
# del alter_ALTERNO_info['ALTERNO_10']

# # print branch
# branch['B_ENDYEAR'] = branch['B_ENDYEAR'].fillna(branch['B_REYEAR'])
# # print(branch['B_ENDYEAR'])
# branch['sub_life'] = branch['B_ENDYEAR'].fillna(branch['B_REYEAR']) - branch['B_REYEAR']
# # 筛选数据
# branch = branch[branch['sub_life']>=0]
# branch_count = branch.groupby(['EID'],as_index=False)['TYPECODE'].count()
# branch_count.rename(columns = {'TYPECODE':'branch_count'},inplace=True)
# branch = pd.merge(branch,branch_count,on=['EID'],how='left')
# branch['branch_count'] = np.log1p(branch['branch_count'])
# branch['branch_count'] = branch['branch_count'].astype(int)
# branch['survive_num']=branch.groupby(['EID'])['sub_life'].count()
# branch['sub_life'] = branch['sub_life'].replace({0.0:-1})
# # print(branch)

# home_prob = branch.groupby(by=['EID'])['IFHOME'].sum()/ branch.groupby(by=['EID'])['IFHOME'].count()
# home_prob = home_prob.reset_index()
# bran_last_year=pd.DataFrame(branch[['EID','B_REYEAR']]).sort_values(['B_REYEAR'],ascending=False).drop_duplicates(['EID'])
# branch = pd.DataFrame(branch[['EID','sub_life']]).drop_duplicates('EID')
# branch = pd.merge(branch,home_prob,on=['EID'],how='left')
# branch = pd.merge(branch,bran_last_year,on=['EID'])

branch['B_ENDYEAR'] = branch['B_ENDYEAR'].fillna(branch['B_REYEAR'])
branch['sub_life'] = branch['B_ENDYEAR']- branch['B_REYEAR']
branch['sub_life']=branch['sub_life']>0
branch['sub_life']=branch['sub_life'].astype(int)
branch_close_rate=branch.groupby(['EID'])['sub_life'].sum()/branch.groupby(['EID'])['sub_life'].count()
branch_close_rate=branch_close_rate.reset_index()
branch_close_rate.rename(columns={'sub_life':'branch_close_rate'})
# 筛选数据
branch_count = branch.groupby(['EID'],as_index=False)['TYPECODE'].count()
branch_count.rename(columns = {'TYPECODE':'branch_count'},inplace=True)
branch_count['branch_count']=branch_count['branch_count'].map(lambda x:normalize1(x,branch_count['branch_count'].max()))

# print(branch_count.head(20))
branch = pd.merge(branch,branch_count,on=['EID'],how='left')
# print(branch.head(20))
branch['branch_count'] = np.log1p(branch['branch_count'])
branch['branch_count'] = branch['branch_count'].astype(int)
branch['sub_life'] = branch['sub_life'].replace({0.0:-1})
branch['sub_life'].fillna(-1)

home_prob = branch.groupby(by=['EID'])['IFHOME'].sum()/ branch.groupby(by=['EID'])['IFHOME'].count()
home_prob = home_prob.reset_index()

bran_first_year=pd.DataFrame(branch[['EID','B_REYEAR']]).sort_values(['B_REYEAR'],ascending=True).drop_duplicates(['EID'])
bran_first_year.rename(columns={'B_REYEAR':'bran_firstY'},inplace=True)
bran_last_year=pd.DataFrame(branch[['EID','B_REYEAR']]).sort_values(['B_REYEAR'],ascending=False).drop_duplicates(['EID'])
bran_last_year.rename(columns={'B_REYEAR':'bran_lastY'},inplace=True)
branch = pd.DataFrame(branch[['EID','branch_count']]).drop_duplicates('EID')
branch = pd.merge(branch,home_prob,on=['EID'],how='left')
branch=pd.merge(branch,branch_close_rate,on=['EID'],how='left')
branch = pd.merge(branch,bran_last_year,on=['EID'],how='left')
branch=pd.merge(branch,bran_first_year,on=['EID'],how='left')
# branch['year_sub']=branch['bran_lastY']-branch['bran_firstY']




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


invest = pd.merge(invest[['EID']],BTBL_INFO,on=['EID'],how='left').drop_duplicates(['EID'])
invest =pd.merge(invest[['EID']],invest_home_prob,on=['EID'],how='left')
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
money_sum['money_sum']=np.log1p(money_sum['money_sum'])

# money_sum=lawsuit[['EID','LAWAMOUNT']].groupby('EID').sum()
# money_sum=money_sum.rename(columns={"LAWAMOUNT":"money_sum"})
# zczb=entbase[['EID','ZCZB']].copy()
# zczb=zczb[zczb['EID'].isin(money_sum.index)]
# zczb=zczb.rename(columns={"ZCZB":"money_sum"})
# zczb=zczb.groupby('EID').max()
# money_BL=money_sum.div(zczb)
# money_BL=money_BL.rename(columns={"money_sum":"money_BL"})
# money_BL['money_BL']=money_BL['money_BL']/10000
# money_BL=money_BL.reset_index()

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
print(train.shape)
print('select sample to train set...')
print(train.shape)

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
# train=pd.merge(train,money_BL,on=['EID'],how='left')

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
# test=pd.merge(test,money_BL,on=['EID'],how='left')
test = test.fillna(-999)
train = train.fillna(-999)

del train['EID']
test_index = test.pop('EID')
# print(train)
print(test_index)


tmp1 = train[train.TARGET==1]
tmp0 = train[train.TARGET==0]
x_valid_1 = tmp1.sample(frac=0.3, random_state=70, axis=0)
x_train_1 = tmp1.drop(x_valid_1.index.tolist())
x_valid_2 = tmp0.sample(frac=0.1, random_state=70, axis=0)
x_train_2 = tmp0.drop(x_valid_2.index.tolist())
X_train = pd.concat([x_train_1,x_train_2],axis=0)

y_train = X_train.pop('TARGET')

X_test = pd.concat([x_valid_1,x_valid_2],axis=0)
y_test = X_test.pop('TARGET')

feature_len = X_train.shape[1]
feature=X_train.columns

X_train = X_train.values
X_test = X_test.values
y_train = y_train.values
y_test = y_test.values

lgb_train = lgb.Dataset(X_train, y_train)
lgb_eval = lgb.Dataset(X_test, y_test, reference=lgb_train)

params={
    'task': 'train',
    'boosting_type': 'gbdt',
    'objective': 'binary',
    'metric': {'auc'},
    'num_leaves': 128,  #128
    'learning_rate': 0.05,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'verbose': 0,
    # 'max_bin':1,

    # 'max_depth':7
}

evals_result = {}
print('Start training...')
# train
gbm = lgb.train(params,
                lgb_train,
                num_boost_round = 1000,
                valid_sets=lgb_eval,
                feature_name=['f' + str(i + 1) for i in range(feature_len)],
                early_stopping_rounds=50,
                evals_result=evals_result)

print('Plot metrics during training...')
ax = lgb.plot_metric(evals_result, metric='auc')
plt.show()

feature_len = X_train.shape[1]
for i in range(feature_len):
    print(i+1,"-->",feature[i])

print('Plot feature importances...')
lgb.plot_importance(gbm,max_num_features=feature_len)
plt.show()



print('Start predicting...')
y_pred = gbm.predict(test.values, num_iteration=gbm.best_iteration)
y_pred = np.round(y_pred,8)
result = pd.DataFrame({'PROB':list(y_pred),})
result['FORTARGET'] = result['PROB'] > 0.222
result['PROB'] = result['PROB'].astype('str')
result['FORTARGET'] = result['FORTARGET'].astype('int')
result = pd.concat([test_index,result],axis=1)
print('positive sample',result[result.FORTARGET == 1].__len__())
print('positive ration',result[result.FORTARGET == 1].__len__() * 1.0/ len(result))
print('predict pos tation',sum(result['FORTARGET']))

result = pd.DataFrame(result).drop_duplicates(['EID'])
result[['EID','FORTARGET','PROB']].to_csv('./evaluation_public.csv',index=None)

print(len(result.EID.unique()))
