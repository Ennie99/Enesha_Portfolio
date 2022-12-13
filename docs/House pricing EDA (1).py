#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import norm
from sklearn.preprocessing import StandardScaler
from scipy import stats
import warnings
import numpy as np # linear algebra
warnings.filterwarnings('ignore')
get_ipython().run_line_magic('matplotlib', 'inline')

# Exploratory data analysis of house prices: Project overview
# 1. Analyzed and identified types of variables
# Examined the essential "SalePrice" variable
# Visualized relationship between 'SalePrice' and key variables
# Identified missing data and deleted not essential data


# In[2]:


#reading the file
df_train = pd.read_csv("C:/Users/Enesha/Downloads/train (1).csv")


# In[3]:


df_train.index


# In[4]:


df_train.head()


# In[5]:


df_train['SalePrice'].describe()


# In[6]:


#histogram 
sns.displot(df_train['SalePrice'], color='purple', height=8, kde=True, aspect=1.5)


# In[7]:


#skewness and kurtosis
print("skewness: %f" % df_train['SalePrice'].skew())#величина отклонения распределения от норм распр.
print("kurtosis: %f" % df_train['SalePrice'].kurt())


# In[8]:


#GrLivArea-above ground living area square feet
#scatter plot grlivare/saleprice
temp = 'GrLivArea'
data = pd.concat([df_train['SalePrice'], df_train[temp]], axis = 1)
data.plot.scatter(x=temp, y='SalePrice', ylim=(0,800000));


# In[9]:


#TotalBsmtSF- Total square feet of basement area
#scatterplot totalbstsf/saleprice
temp = 'TotalBsmtSF'
data = pd.concat([df_train['SalePrice'], df_train[temp]], axis = 1)
data.plot.scatter(x=temp, y='SalePrice', ylim=(0,800000));


# In[10]:


#box plot overallquality/saleprice
temp = 'OverallQual'
data = pd.concat([df_train['SalePrice'], df_train[temp]], axis = 1)
plt.figure(figsize=(8, 6))
fig = sns.boxplot(x=temp, y='SalePrice', data=data)
fig.axis(ymin=0,ymax=800000);


# In[11]:


#correlation matrix
corrmat = df_train.corr()
plt.figure(figsize=(12, 9))
sns.heatmap(corrmat,vmax=.8, square=True)


# In[12]:


#scatterplot
sns.set()
cols = ['SalePrice', 'OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', 'FullBath', 'YearBuilt']
sns.pairplot(df_train[cols], size=2.5)
plt.show();


# In[13]:


#the missing data
total = df_train.isnull().sum().sort_values(ascending=False)
percent = (df_train.isnull().sum()/df_train.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total,percent], axis=1, keys=["total", "percent"])
missing_data.head(20)


# In[ ]:





# In[14]:


#deleting columns with missing values
df_train.dropna(axis='columns', how='any',inplace=True)


# In[ ]:





# In[15]:


#dealing with the missing data
#df_train1 = df_train.drop(labels=['PoolQC', 'MiscFeature', 'Alley', 'Fence', 'FireplaceQu', 'LotFrontage', \
 #                                          'GarageYrBlt', 'GarageCond', 'GarageType', 'GarageFinish', 'GarageQual', 'BsmtFinType2', 'BsmtExposure', \
 #                                          'BsmtQual','BsmtCond', 'BsmtFinType1', 'MasVnrArea', 'MasVnrType'], axis=1)
#df_train = df_train.drop(df_train.loc[df_train['Electrical'].isnull()].index)


# In[15]:


#checking if there are any columns with missing values
df_train.isnull().sum().max()


# In[16]:


#standardizing data
saleprice_scaled = StandardScaler().fit_transform(df_train['SalePrice'][:,np.newaxis]);
low_range = saleprice_scaled[saleprice_scaled[:,0].argsort()][:10]
high_range = saleprice_scaled[saleprice_scaled[:,0].argsort()][-10:]
print('outer range(low) of the distribution')
print(low_range)
print('outer range(high) of the distribution')
print(high_range)


# In[17]:


temp = 'GrLivArea'
data = pd.concat([df_train['SalePrice'], df_train[temp]], axis=1)
data.plot.scatter(x=temp,y='SalePrice',ylim=(0,800000));


# In[18]:


df_train.sort_values(by = 'GrLivArea', ascending=False)[:2]


# In[19]:


df_train = df_train.drop(df_train[df_train['Id'] == 1299].index)
df_train = df_train.drop(df_train[df_train['Id'] == 524].index)


# In[20]:


#bivariate analysis saleprice/grlivarea
var = 'TotalBsmtSF'
data = pd.concat([df_train['SalePrice'], df_train[var]], axis=1)
data.plot.scatter(x=var, y='SalePrice', ylim=(0,800000));


# In[21]:


df_train.sort_values(by = 'TotalBsmtSF', ascending=False)[:1]


# In[22]:


df_train = df_train.drop(df_train[df_train['Id'] == 1299].index)


# In[23]:


data.plot.scatter(x=var, y='SalePrice', ylim=(0,800000));


# In[24]:


#histogram and probability plot
sns.distplot(df_train['SalePrice'], fit=norm)
fig = plt.figure()
result = stats.probplot(df_train['SalePrice'], plot=plt)


# In[26]:


# our plot has positive skewness, so we can use log transformation 

df_train['SalePrice'] = np.log(df_train['SalePrice'])

sns.distplot(df_train['SalePrice'], fit=norm)
fig = plt.figure()
result = stats.probplot(df_train['SalePrice'], plot=plt)


# In[28]:


#the same goes for grlivarea variable
sns.distplot(df_train['GrLivArea'], fit=norm)
fig = plt.figure()
result = stats.probplot(df_train['GrLivArea'], plot=plt)


# In[29]:


#data transformation
df_train['GrLivArea'] = np.log(df_train['GrLivArea'])

sns.distplot(df_train['GrLivArea'], fit=norm);
fig = plt.figure()
result = stats.probplot(df_train['GrLivArea'], plot=plt)


# In[30]:


sns.distplot(df_train['TotalBsmtSF'], fit=norm);
fig = plt.figure()
res = stats.probplot(df_train['TotalBsmtSF'], plot=plt)


# In[33]:


#because we have a lot of houses with no basement, it means we cannot apply log transformation
# so i will create Series which contains only houses with basement,ignoring the ones that don't have it

df_train['HasBsmt'] = pd.Series(len(df_train['TotalBsmtSF']), index=df_train.index)
df_train['HasBsmt'] = 0
df_train.loc[df_train['TotalBsmtSF']>0, 'HasBsmt'] = 1


# In[37]:


#log transformation
df_train.loc[df_train['HasBsmt']==1, 'TotalBsmtSF'] = np.log(df_train['TotalBsmtSF'])


# In[38]:


sns.distplot(df_train[df_train['TotalBsmtSF']>0]['TotalBsmtSF'], fit=norm);
fig = plt.figure()
res = stats.probplot(df_train[df_train['TotalBsmtSF']>0]['TotalBsmtSF'], plot=plt)


# In[39]:


#scatter plot
plt.scatter(df_train['GrLivArea'], df_train['SalePrice']);


# In[40]:


#scatter plot
plt.scatter(df_train[df_train['TotalBsmtSF']>0]['TotalBsmtSF'], df_train[df_train['TotalBsmtSF']>0]['SalePrice']);


# In[41]:


#convert categorical variable into dummy
df_train = pd.get_dummies(df_train)


# In[ ]:




