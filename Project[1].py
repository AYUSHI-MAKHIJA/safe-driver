#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE
from numpy.random import RandomState
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict, StratifiedKFold
import warnings
warnings.filterwarnings('ignore')

get_ipython().run_line_magic('matplotlib', 'inline')
sns.set_palette(['#06B1F0', '#FC4B60'])
random_seed = 63445


# In[2]:


df = pd.read_csv("train.csv")
df.columns


# In[3]:


df.info()


# In[4]:


(df==-1).sum()


# # Filling missing values for Numerical Features

# In[5]:


df['ps_reg_03'].mean()


# In[6]:


df['ps_reg_03']=df['ps_reg_03'].replace(-1,np.NAN)
df['ps_reg_03'].mean()


# In[7]:


df['ps_reg_03']=df['ps_reg_03'].replace(np.nan,df['ps_reg_03'].mean())
df['ps_reg_03'].mean()


# In[8]:


ax = sns.distplot(df.ps_reg_03)
plt.show()


# In[9]:


imputer = SimpleImputer(missing_values = -1 , strategy = 'most_frequent')
df['ps_car_11'] = imputer.fit_transform(df[['ps_car_11']])
(df['ps_car_11']==-1).sum()


# In[10]:


sns.countplot(df.ps_car_11);
plt.xlabel('ps_car_11');
plt.ylabel('y');
plt.show()


# In[11]:


df['ps_car_14']=df['ps_car_14'].replace(-1,np.NAN)
df['ps_car_14']=df['ps_car_14'].replace(np.nan,df['ps_car_14'].mean())


# In[12]:


ax = sns.distplot(df.ps_car_14)
plt.show()


# In[ ]:





# # Filling missing values of categorical features

# In[22]:


imputer = SimpleImputer(missing_values = -1 , strategy = 'most_frequent')
df['ps_ind_02_cat'] = imputer.fit_transform(df[['ps_ind_02_cat']])
(df['ps_ind_02_cat']==-1).sum()


# In[23]:


sns.countplot(df.ps_ind_02_cat);
plt.xlabel('ps_ind_02_cat');
plt.ylabel('y');
plt.show()


# In[24]:


imputer = SimpleImputer(missing_values = -1 , strategy = 'most_frequent')
df['ps_ind_04_cat'] = imputer.fit_transform(df[['ps_ind_04_cat']])
(df['ps_ind_04_cat']==-1).sum()


# In[25]:


sns.countplot(df.ps_ind_04_cat);
plt.xlabel('ps_ind_04_cat');
plt.ylabel('y');
plt.show()


# In[26]:


imputer = SimpleImputer(missing_values = -1 , strategy = 'most_frequent')
df['ps_ind_05_cat'] = imputer.fit_transform(df[['ps_ind_05_cat']])
(df['ps_ind_05_cat']==-1).sum()


# In[27]:


sns.countplot(df.ps_ind_05_cat);
plt.xlabel('ps_ind_05_cat');
plt.ylabel('y');
plt.show()


# In[28]:


imputer = SimpleImputer(missing_values = -1 , strategy = 'median')
df['ps_car_01_cat'] = imputer.fit_transform(df[['ps_car_01_cat']])
(df['ps_car_01_cat']==-1).sum()


# In[29]:


sns.countplot(df.ps_car_01_cat);
plt.xlabel('ps_car_01_cat');
plt.ylabel('y');
plt.show()


# In[30]:


imputer = SimpleImputer(missing_values = -1 , strategy = 'most_frequent')
df['ps_car_02_cat'] = imputer.fit_transform(df[['ps_car_02_cat']])
(df['ps_car_02_cat']==-1).sum()


# In[31]:


sns.countplot(df.ps_car_02_cat);
plt.xlabel('ps_car_02_cat');
plt.ylabel('y');
plt.show()


# In[32]:


df['ps_car_03_cat']=df['ps_car_03_cat'].replace(-1,np.NAN)


# In[33]:


df['ps_car_03_cat']=df['ps_car_03_cat'].fillna(method='bfill')


# In[34]:


sns.countplot(df.ps_car_03_cat);
plt.xlabel('ps_car_03_cat');
plt.ylabel('y');
plt.show()


# In[35]:


df['ps_car_05_cat']=df['ps_car_05_cat'].replace(-1,np.NAN)


# In[36]:


df['ps_car_05_cat']=df['ps_car_05_cat'].fillna(method='bfill')


# In[37]:


sns.countplot(df.ps_car_05_cat );
plt.xlabel('ps_car_05_cat?');
plt.ylabel('Number of occurrences');
plt.show()


# In[38]:


df['ps_car_09_cat'].value_counts()


# In[39]:


df['ps_car_07_cat'].value_counts()


# In[40]:


df['ps_car_09_cat'] = df['ps_car_09_cat'].replace(-1, df['ps_car_09_cat'].value_counts().index[0])
df['ps_car_07_cat'] = df['ps_car_07_cat'].replace(-1, df['ps_car_07_cat'].value_counts().index[0])


# In[41]:


df['ps_car_09_cat'].value_counts()


# In[42]:


df['ps_car_07_cat'].value_counts()


# In[43]:


(df == -1).sum()
#df.isnull().sum()


# In[44]:


fd = pd.read_csv("test.csv")
fd.columns


# In[45]:


(fd==-1).sum()


# In[46]:


fd.isnull().sum()


# In[47]:


imputer = SimpleImputer(missing_values = -1 , strategy = 'most_frequent')
fd['ps_car_02_cat'] = imputer.fit_transform(fd[['ps_car_02_cat']])
(fd['ps_car_02_cat']==-1).sum()


# In[48]:


x_test.isnull().sum()


# In[49]:


sns.countplot(fd.ps_car_02_cat);
plt.xlabel('ps_car_02_cat');
plt.ylabel('y');
plt.show()


# In[50]:


imputer = SimpleImputer(missing_values = np.NAN , strategy = 'most_frequent')
fd['ps_car_03_cat'] = imputer.fit_transform(fd[['ps_car_03_cat']])
(fd['ps_car_03_cat']==np.NAN).sum()


# In[51]:


imputer = SimpleImputer(missing_values = np.NAN , strategy = 'most_frequent')
fd['ps_car_05_cat'] = imputer.fit_transform(fd[['ps_car_05_cat']])
(fd['ps_car_05_cat']==np.NAN).sum()


# In[52]:


imputer = SimpleImputer(missing_values = -1 , strategy = 'most_frequent')
fd['ps_car_04_cat'] = imputer.fit_transform(fd[['ps_car_04_cat']])
(fd['ps_car_04_cat']==-1).sum()


# In[53]:


sns.countplot(fd.ps_car_04_cat);
plt.xlabel('ps_car_04_cat');
plt.ylabel('y');
plt.show()


# In[54]:


imputer = SimpleImputer(missing_values = -1 , strategy = 'most_frequent')
fd['ps_car_04_cat'] = imputer.fit_transform(fd[['ps_car_04_cat']])
(fd['ps_car_04_cat']==-1).sum()


# In[55]:


sns.countplot(fd.ps_car_04_cat);
plt.xlabel('ps_car_04_cat');
plt.ylabel('y');
plt.show()


# In[56]:


fd['ps_car_01_cat']=fd['ps_car_01_cat'].replace(-1,np.NAN)


# In[57]:


fd['ps_car_01_cat']=fd['ps_car_01_cat'].fillna(method='bfill')


# In[58]:


sns.countplot(fd.ps_car_01_cat);
plt.xlabel('ps_car_01_car');
plt.ylabel('y');
plt.show()


# In[59]:


fd['ps_car_03_cat']=fd['ps_car_03_cat'].replace(-1,np.NAN)


# In[60]:


fd['ps_car_03_cat']=fd['ps_car_03_cat'].fillna(method='bfill')


# In[61]:


sns.countplot(fd.ps_car_03_cat);
plt.xlabel('ps_car_03_cat');
plt.ylabel('y');
plt.show()


# In[62]:


fd['ps_car_05_cat']=fd['ps_car_05_cat'].replace(-1,np.NAN)


# In[63]:


fd['ps_car_05_cat']=fd['ps_car_05_cat'].fillna(method='bfill')


# In[64]:


sns.countplot(fd.ps_car_05_cat);
plt.xlabel('ps_car_05_cat');
plt.ylabel('y');
plt.show()


# In[65]:


fd['ps_car_07_cat']=fd['ps_car_07_cat'].replace(-1,np.NAN)


# In[66]:


fd['ps_car_07_cat']=fd['ps_car_07_cat'].fillna(method='bfill')


# In[67]:


sns.countplot(fd.ps_car_07_cat);
plt.xlabel('ps_car_07_cat');
plt.ylabel('y');
plt.show()


# In[68]:


fd['ps_car_09_cat']=fd['ps_car_09_cat'].replace(-1,np.NAN)


# In[69]:


fd['ps_car_09_cat']=fd['ps_car_09_cat'].fillna(method='bfill')


# In[70]:


sns.countplot(fd.ps_car_09_cat);
plt.xlabel('ps_car_09_cat');
plt.ylabel('y');
plt.show()


# In[71]:


fd['ps_reg_03']=fd['ps_reg_03'].replace(-1,np.NAN)
fd['ps_reg_03'].mean()


# In[72]:


fd['ps_reg_03']=fd['ps_reg_03'].replace(np.nan,fd['ps_reg_03'].mean())
fd['ps_reg_03'].mean()


# In[73]:


imputer = SimpleImputer(missing_values = -1 , strategy = 'most_frequent')
fd['ps_car_11'] = imputer.fit_transform(fd[['ps_car_11']])
(fd['ps_car_11']==-1).sum()


# In[74]:


sns.countplot(fd.ps_car_11);
plt.xlabel('ps_car_11');
plt.ylabel('y');
plt.show()


# In[75]:


imputer = SimpleImputer(missing_values = -1 , strategy = 'most_frequent')
fd['ps_car_12'] = imputer.fit_transform(fd[['ps_car_12']])
(fd['ps_car_12']==-1).sum()


# In[76]:


sns.countplot(fd.ps_car_12);
plt.xlabel('ps_car_12');
plt.ylabel('y');
plt.show()


# In[77]:


fd['ps_car_14']=fd['ps_car_14'].replace(-1,np.NAN)
fd['ps_car_14']=fd['ps_car_14'].replace(np.nan,fd['ps_car_14'].mean())


# In[78]:


sns.countplot(fd.ps_car_14);
plt.xlabel('ps_car_14');
plt.ylabel('y');
plt.show()


# In[79]:


imputer = SimpleImputer(missing_values = -1 , strategy = 'most_frequent')
fd['ps_ind_02_cat'] = imputer.fit_transform(fd[['ps_ind_02_cat']])
(fd['ps_ind_02_cat']==-1).sum()


# In[80]:


sns.countplot(fd.ps_ind_02_cat);
plt.xlabel('ps_ind_02_cat');
plt.ylabel('y');
plt.show()


# In[81]:


imputer = SimpleImputer(missing_values = -1 , strategy = 'most_frequent')
fd['ps_ind_04_cat'] = imputer.fit_transform(fd[['ps_ind_04_cat']])
(fd['ps_ind_04_cat']==-1).sum()


# In[82]:


sns.countplot(fd.ps_ind_04_cat);
plt.xlabel('ps_ind_04_cat');
plt.ylabel('y');
plt.show()


# In[83]:


imputer = SimpleImputer(missing_values = -1 , strategy = 'most_frequent')
fd['ps_ind_05_cat'] = imputer.fit_transform(fd[['ps_ind_05_cat']])
(fd['ps_ind_05_cat']==-1).sum()


# In[84]:


sns.countplot(fd.ps_ind_05_cat);
plt.xlabel('ps_ind_05_cat');
plt.ylabel('y');
plt.show()


# In[85]:


(fd==-1).sum()


# In[86]:


(fd==np.NaN).sum()


# In[87]:


df = df.reset_index()
fd = fd.reset_index()


# In[88]:


from sklearn.feature_selection import SelectKBest , f_classif,SelectPercentile
from sklearn.feature_selection import chi2


# # Assiging the value in x_train, x_test,y_train

# In[89]:


y_train  =  df.target
df.drop(['target', 'id'], axis='columns', inplace=True)
df.drop('index',inplace=True,axis='columns')
x_train = df
fd.drop('id',inplace=True,axis='columns')


# In[90]:


fd.drop('index',inplace=True,axis='columns')


# In[91]:


x_test = fd


# In[92]:


x_test.shape


# In[93]:


fd.isnull().sum()


# In[ ]:





# In[95]:


imputer = SimpleImputer(missing_values = np.NaN , strategy = 'most_frequent')
x_test['ps_car_03_cat'] = imputer.fit_transform(x_test[['ps_car_03_cat']])
(x_test['ps_car_03_cat']==np.NaN).sum()


# In[96]:


imputer = SimpleImputer(missing_values = np.NaN , strategy = 'most_frequent')
x_test['ps_car_05_cat'] = imputer.fit_transform(x_test[['ps_car_05_cat']])
(x_test['ps_car_05_cat']==-1).sum()


# # Feature Selection using SelectKBest

# selector = SelectKBest(f_classif, k=20)
# selector.fit(x_train, y_train)
# x_train_selected = selector.transform(x_train)

# In[101]:


print("X_train.shape: {}".format(x_train.shape))
print("X_train_selected.shape: {}".format(x_train_selected.shape))


# In[102]:


x_train.shape


# In[103]:


x_train.columns


# In[104]:


x_test.shape


# In[105]:


(x_test==-1).sum()


# In[106]:


(x_test == np.NaN).sum()


# In[107]:


t=np.any(np.isnan(x_train)) 
print(t)


# In[108]:


t=np.any(np.isnan(x_test)) 
print(t)


# In[109]:


t=np.all(np.isfinite(x_train))
print(t)


# In[110]:


t=np.all(np.isfinite(x_test))
print(t)


# # APPLYING LOGISTIC REGRESSION

# In[113]:


from sklearn.linear_model import LogisticRegression
# transform test data
x_test_selected = selector.transform(x_test)
lr = LogisticRegression()
lr.fit(x_train_selected, y_train)


# # with the help of getsupport we can find out which features are selected by the algorithm

# In[112]:


mask = selector.get_support()
print(mask)
# visualize the mask -- black is True, white is False
plt.matshow(mask.reshape(1, -1), cmap='gray_r')
plt.xlabel("Sample index")


# In[121]:


preds = lr.predict(x_test_selected)
print(preds)


# # Feature Selection using SelectPercentile

# In[851]:


select = SelectPercentile(percentile=50)
select.fit(x_train, y_train)
# transform training set
x_train_selected = select.transform(x_train)
print("x_train.shape: {}".format(x_train.shape))
print("x_train_selected.shape: {}".format(x_train_selected.shape))


# In[856]:


from sklearn.linear_model import LogisticRegression
# transform test data
x_test_selected = select.transform(x_test)


# In[858]:


mask = select.get_support()
print(mask)
# visualize the mask -- black is True, white is False
plt.matshow(mask.reshape(1, -1), cmap='gray_r')
plt.xlabel("Sample index")


# # Feature_Selection using RFE(Recursive Feature eliminator)

# In[862]:


from sklearn.feature_selection import RFE


# In[ ]:


#select = RFE(RandomForestClassifier(n_estimators=2, random_state=42),n_features_to_select=8)
#select.fit(x_train, y_train)
# visualize the selected features:
#mask = select.get_support()
#plt.matshow(mask.reshape(1, -1), cmap='gray_r')
#plt.xlabel("Sample index")


# In[ ]:


#x_train_rfe= select.transform(x_train)
#x_test_rfe= select.transform(x_test)
#score = LogisticRegression().fit(x_train_rfe, y_train).score(x_test_rfe, y_test)
#print("Test score: {:.3f}".format(score))


# In[ ]:




