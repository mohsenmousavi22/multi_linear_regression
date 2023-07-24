#!/usr/bin/env python
# coding: utf-8

# # Multiple Linear Regression

# ## Importing the libraries

# In[20]:


import numpy as np
import pandas as pd


# ## Importing the dataset

# In[21]:


dataset = pd.read_csv('50_Startups.csv')
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values


# In[22]:


print(x)


# In[23]:


print(y)


# ## Encoding categorical data

# In[24]:


from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [3])], remainder='passthrough')
x = np.array(ct.fit_transform(x))


# In[25]:


print(x)


# ## Splitting the dataset into the Training set and Test set

# In[26]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 1)


# In[27]:


print(x_train)


# In[28]:


print(y_train)


# In[29]:


print(x_test)


# In[30]:


print(y_test)


# ## Training the Multiple Linear Regression model on the Training set

# In[31]:


from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train, y_train)


# ## Predicting the Test set results

# In[33]:


y_pred = regressor.predict(x_test)
np.set_printoptions(precision = 2)
print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)), 1))

