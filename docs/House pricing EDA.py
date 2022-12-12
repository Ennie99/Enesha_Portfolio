#!/usr/bin/env python
# coding: utf-8

# In[9]:


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


# In[14]:


#reading the file
df_train = pd.read_csv("C:/Users/Enesha/Downloads/train (1).csv")


# In[15]:


df_train.index


# In[16]:


df_train.head()


# In[17]:


df_train['SalePrice'].describe()


# In[18]:


#histogram 
sns.displot(df_train['SalePrice'], color='purple', height=8, kde=True, aspect=1.5)


# In[19]:


#skewness and kurtosis
print("skewness: %f" % df_train['SalePrice'].skew())#величина отклонения распределения от норм распр.
print("kurtosis: %f" % df_train['SalePrice'].kurt())


# In[20]:


#GrLivArea-above ground living area square feet
#scatter plot grlivare/saleprice
temp = 'GrLivArea'
data = pd.concat([df_train['SalePrice'], df_train[temp]], axis = 1)
data.plot.scatter(x=temp, y='SalePrice', ylim=(0,800000));


# In[21]:


#TotalBsmtSF- Total square feet of basement area
#scatterplot totalbstsf/saleprice
temp = 'TotalBsmtSF'
data = pd.concat([df_train['SalePrice'], df_train[temp]], axis = 1)
data.plot.scatter(x=temp, y='SalePrice', ylim=(0,800000));


# In[22]:


#box plot overallquality/saleprice
temp = 'OverallQual'
data = pd.concat([df_train['SalePrice'], df_train[temp]], axis = 1)
plt.figure(figsize=(8, 6))
fig = sns.boxplot(x=temp, y='SalePrice', data=data)
fig.axis(ymin=0,ymax=800000);


# In[23]:


#correlation matrix
corrmat = df_train.corr()
plt.figure(figsize=(12, 9))
sns.heatmap(corrmat,vmax=.8, square=True)


# In[24]:


#scatterplot
sns.set()
cols = ['SalePrice', 'OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', 'FullBath', 'YearBuilt']
sns.pairplot(df_train[cols], size=2.5)
plt.show();


# In[25]:


#the missing data
total = df_train.isnull().sum().sort_values(ascending=False)
percent = (df_train.isnull().sum()/df_train.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total,percent], axis=1, keys=["total", "percent"])
missing_data.head(20)


# In[ ]:





# In[26]:


#deleting columns with missing values
df_train.dropna(axis='columns', how='any',inplace=True)


# In[ ]:





# In[15]:


#dealing with the missing data
#df_train1 = df_train.drop(labels=['PoolQC', 'MiscFeature', 'Alley', 'Fence', 'FireplaceQu', 'LotFrontage', \
 #                                          'GarageYrBlt', 'GarageCond', 'GarageType', 'GarageFinish', 'GarageQual', 'BsmtFinType2', 'BsmtExposure', \
 #                                          'BsmtQual','BsmtCond', 'BsmtFinType1', 'MasVnrArea', 'MasVnrType'], axis=1)
#df_train = df_train.drop(df_train.loc[df_train['Electrical'].isnull()].index)


# In[28]:


#checking if there are any columns with missing values
df_train.isnull().sum().max()


# In[31]:


#standardizing data
saleprice_scaled = StandardScaler().fit_transform(df_train['SalePrice'][:,np.newaxis]);
low_range = saleprice_scaled[saleprice_scaled[:,0].argsort()][:10]
high_range = saleprice_scaled[saleprice_scaled[:,0].argsort()][-10:]
print('outer range(low) of the distribution')
print(low_range)
print('outer range(high) of the distribution')
print(high_range)


# In[32]:


temp = 'GrLivArea'
data = pd.concat([df_train['SalePrice'], df_train[temp]], axis=1)
data.plot.scatter(x=temp,y='SalePrice',ylim=(0,800000));


# In[35]:


df_train.sort_values(by = 'GrLivArea', ascending=False)[:2]


# In[43]:


df_train = df_train.drop(df_train[df_train['Id'] == 1299].index)
df_train = df_train.drop(df_train[df_train['Id'] == 524].index)


# In[44]:


#bivariate analysis saleprice/grlivarea
var = 'TotalBsmtSF'
data = pd.concat([df_train['SalePrice'], df_train[var]], axis=1)
data.plot.scatter(x=var, y='SalePrice', ylim=(0,800000));


# In[45]:


df_train.sort_values(by = 'TotalBsmtSF', ascending=False)[:1]


# In[46]:


df_train = df_train.drop(df_train[df_train['Id'] == 1299].index)


# In[47]:


data.plot.scatter(x=var, y='SalePrice', ylim=(0,800000));


# In[ ]:




