
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd


# In[2]:


titanic_df = pd.read_csv("C:/Users/tewat/DataSciense/kaggleData/train.csv")


# In[3]:


titanic_df.head()


# In[4]:


titanic_df_age = titanic_df[(titanic_df['Age']>35)& (titanic_df['Age']<50) ]


# In[5]:


titanic_df_age.head(5)


# In[6]:


#object is categorical data
titanic_df['Embarked'].value_counts()


# In[7]:


titanic_df['Embarked'].unique()


# In[8]:


titanic_df['Embarked'] = titanic_df['Embarked'].fillna('S')


# In[9]:


titanic_df['Embarked'].value_counts()


# In[10]:


titanic_df['Embarked'] = titanic_df['Embarked'].fillna("S")


# In[11]:


titanic_df['Embarked'].value_counts()


# In[12]:


titanic_df=titanic_df['Age'].isnull().dropna(how='any')


# In[13]:


titanic_df.head()


# In[17]:


#checking iris data null values
iris_data_df = pd.read_csv("C:/Users/tewat/DataSciense/iris.data.txt")                           


# In[24]:


iris_data_df


# In[20]:


iris_data_df.shape


# In[34]:


iris_data_df[(iris_data_df['5.1'] > 3.0) & (iris_data_df['5.1'] < 4.1)]


# In[25]:


iris_data_df.dtypes


# In[27]:


iris_data_df['Iris-setosa'].value_counts()


# In[29]:


iris_data_df['Iris-setosa'].isnull().value_counts()


# In[30]:


iris_data_df['Iris-setosa'].unique()


# In[31]:


len(iris_data_df['Iris-setosa'].unique())


# In[32]:


iris_data_df=iris_data_df.dropna(how='any')


# In[33]:


len(iris_data_df['Iris-setosa'].unique())

