#!/usr/bin/env python
# coding: utf-8

# In[23]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')


# In[24]:


df = pd.read_csv("train.csv")
df.head()
df.shape


# In[25]:


df['ps_reg_03'].mean()


# In[26]:


from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values = -1 , strategy = 'mean')
df['ps_reg_03'] = imputer.fit_transform(df[['ps_reg_03']])
(df['ps_reg_03']== -1).sum()


# In[27]:


df['ps_reg_03'].mean()


# In[28]:


ax = sns.distplot(df.ps_reg_03)
plt.show()


# In[29]:


imputer = SimpleImputer(missing_values = -1 , strategy = 'most_frequent')
df['ps_car_11'] = imputer.fit_transform(df[['ps_car_11']])
(df['ps_car_11']==-1).sum()


# In[30]:


sns.countplot(df.ps_car_11);
plt.xlabel('ps_car_11');
plt.ylabel('y');
plt.show()


# In[31]:


imputer = SimpleImputer(missing_values = -1 , strategy = 'most_frequent')
df['ps_ind_02_cat'] = imputer.fit_transform(df[['ps_ind_02_cat']])
(df['ps_ind_02_cat']==-1).sum()


# In[32]:


sns.countplot(df.ps_ind_02_cat);
plt.xlabel('ps_ind_02_cat');
plt.ylabel('y');
plt.show()


# In[33]:


imputer = SimpleImputer(missing_values = -1 , strategy = 'most_frequent')
df['ps_ind_04_cat'] = imputer.fit_transform(df[['ps_ind_04_cat']])
(df['ps_ind_04_cat']==-1).sum()


# In[34]:


sns.countplot(df.ps_ind_04_cat);
plt.xlabel('ps_ind_04_cat');
plt.ylabel('y');
plt.show()


# In[35]:


imputer = SimpleImputer(missing_values = -1 , strategy = 'most_frequent')
df['ps_ind_05_cat'] = imputer.fit_transform(df[['ps_ind_05_cat']])
(df['ps_ind_05_cat']==-1).sum()


# In[36]:


sns.countplot(df.ps_ind_05_cat);
plt.xlabel('ps_ind_05_cat');
plt.ylabel('y');
plt.show()


# In[37]:


imputer = SimpleImputer(missing_values = -1 , strategy = 'median')
df['ps_car_01_cat'] = imputer.fit_transform(df[['ps_car_01_cat']])
(df['ps_car_01_cat']==-1).sum()


# In[38]:


sns.countplot(df.ps_car_01_cat);
plt.xlabel('ps_car_01_cat');
plt.ylabel('y');
plt.show()


# In[39]:


sns.countplot(df.ps_car_02_cat);
plt.xlabel('ps_car_02_cat');
plt.ylabel('y');
plt.show()


# In[40]:


imputer = SimpleImputer(missing_values = -1 , strategy = 'most_frequent')
df['ps_car_02_cat'] = imputer.fit_transform(df[['ps_car_02_cat']])
(df['ps_car_02_cat']==-1).sum()


# In[41]:


sns.countplot(df.ps_car_02_cat);
plt.xlabel('ps_car_02_cat');
plt.ylabel('y');
plt.show()


# In[ ]:





# In[ ]:




