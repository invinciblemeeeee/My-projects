#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np  
import matplotlib.pyplot as plt  
import seaborn as sns


# In[3]:


url = "http://bit.ly/w-data"
s_data = pd.read_csv(url)
print('Shape of the dataset is:',s_data.shape)
s_data.head()


# In[4]:


s_data.isnull().sum()


# In[5]:


s_data.describe()


# In[6]:


s_data.info()


# In[7]:


s_data.plot(kind='scatter',x='Hours',y='Scores')
plt.show()


# In[8]:


s_data.corr(method='pearson')


# In[9]:


s_data.corr(method='spearman')


# In[10]:


hours=s_data['Hours'] 
scores=s_data['Scores']


# In[12]:


sns.distplot(hours)


# In[13]:


sns.distplot(scores)


# LINEAR REGRESSION

# In[14]:


X = s_data.iloc[:, :-1].values  
y = s_data.iloc[:, 1].values  


# In[15]:


from sklearn.model_selection import train_test_split  
X_train, X_test, y_train, y_test = train_test_split(X, y, 
                            test_size=0.2, random_state=0) 


# In[16]:


from sklearn.linear_model import LinearRegression  
reg = LinearRegression()  
reg.fit(X_train, y_train) 


# In[17]:


m=reg.coef_
c=reg.intercept_
line=m*X+c
plt.scatter(X,y)
plt.plot(X,line);
plt.show()


# In[18]:


y_pred=reg.predict(X_test)


# In[21]:


actual_predicted=pd.DataFrame({'Target':y_test,'Predicted':y_pred})
actual_predicted


# In[25]:


sns.set_style('whitegrid')
sns.distplot(np.array(y_test-y_pred))
plt.show()


# In[26]:


h=9.25
s=reg.predict([[h]])
print("If a student studies for {} hours per day he\she will score {} % in exam.".format(h,s))


# In[27]:


from sklearn import metrics 
from sklearn.metrics import r2_score
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred)) 
print('R2 Score:',r2_score(y_test,y_pred))


# In[ ]:




