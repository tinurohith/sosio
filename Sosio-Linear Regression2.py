
# coding: utf-8

# In[1]:


import os


# In[2]:


import pandas as pd
from pandas import DataFrame
import numpy as np


# In[3]:


import urllib
from urllib.request import urlretrieve


# In[4]:


url="https://github.com/swapniljariwala/nsepy"


# In[5]:


from nsepy import get_history
from datetime import date


# In[6]:


nse_tcs=get_history(symbol="TCS",start=date(2015,1,1),end=date(2015,12,31))


# In[7]:


nse_tcs.columns


# In[8]:


nse_infy=get_history(symbol="INFY",start=date(2015,1,1),end=date(2015,12,31))


# In[9]:


nse_infy.columns


# In[10]:


nse_infy.head()


# In[11]:


nse_tcs.head()


# In[12]:


nse_infy.shape


# In[13]:


nse_tcs.shape


# In[14]:


nse_infy.describe()


# In[15]:


nse_tcs.describe()


# In[16]:


nse_infy.isnull().sum()


# In[17]:


nse_tcs.isnull().sum()


# In[18]:


nse_infy.info()


# In[19]:


nse_tcs.info()


# In[20]:


def movingaverage(x,w):
   return pd.Series(x.rolling(window=w,min_periods=0).mean())


# In[21]:


nse_infy['4weeks']=movingaverage(nse_infy['Close'],20)
nse_infy['16weeks']=movingaverage(nse_infy['Close'],80)
nse_infy['28weeks']=movingaverage(nse_infy['Close'],140)
nse_infy['40weeks']=movingaverage(nse_infy['Close'],200)
nse_infy['52weeks']=movingaverage(nse_infy['Close'],260)
nse_tcs['4weeks']=movingaverage(nse_tcs['Close'],20)
nse_tcs['16weeks']=movingaverage(nse_tcs['Close'],80)
nse_tcs['28weeks']=movingaverage(nse_tcs['Close'],140)
nse_tcs['40weeks']=movingaverage(nse_tcs['Close'],200)
nse_tcs['52weeks']=movingaverage(nse_tcs['Close'],260)


# In[22]:


nse_infy[['Close','4weeks','16weeks','28weeks','40weeks','52weeks']].tail()


# In[23]:


nse_tcs[['Close','4weeks','16weeks','28weeks','40weeks','52weeks']].tail()


# In[24]:


nse_infy.tail()


# In[25]:


nse_tcs.tail()


# In[26]:


def volumeshocks(data):
    data['PreviousVolume']=data['Volume'].shift(1)
    data['VolumeShocks'] = (data['Volume']>data['PreviousVolume']*0.1+data['PreviousVolume']).map({True:0,False:1})
    return data


# In[27]:


nse_infy=volumeshocks(nse_infy)
nse_tcs=volumeshocks(nse_tcs)


# In[28]:


nse_infy[['Volume','PreviousVolume','VolumeShocks']].head()


# In[29]:


nse_tcs[['Volume','PreviousVolume','VolumeShocks']].head()


# In[30]:


def priceshocks(data):
    data['T']=data['Close'].shift(1)
    data['PriceShocks'] = (data['Close']-data['T']>0.20*(data['Close']-data['T'])).map({True:0,False:1})
    return data


# In[31]:


nse_infy=priceshocks(nse_infy)
nse_tcs=priceshocks(nse_tcs)


# In[32]:


nse_infy[['Close','PriceShocks']].head()


# In[33]:


nse_infy[['Close','PriceShocks']].head()


# In[34]:


def blackswan(data):
    data['T1']=data['Prev Close'].shift(1)
    data['BlackSwanPrice'] = (data['Prev Close']-data['T1']>0.20*(data['Prev Close']-data['T1'])).map({True:0,False:1})
    return data


# In[35]:


nse_infy=blackswan(nse_infy)
nse_tcs=blackswan(nse_tcs)


# In[36]:


nse_infy[['Prev Close','BlackSwanPrice']].head()


# In[37]:


nse_tcs[['Prev Close','BlackSwanPrice']].head()


# In[38]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns


# In[39]:


nse_infy.Close.plot(figsize=(20,10),linewidth=5,fontsize=20,grid=True)
plt.title("Close price of INFY")


# In[40]:


nse_tcs.Close.plot(figsize=(20,10),linewidth=5,fontsize=20,grid=True)
plt.title("Close price of TCS")


# In[41]:


nse_infy.Close.plot(figsize=(20,10),linewidth=5,fontsize=20,grid=True)
nse_tcs.Close.plot(figsize=(20,10),linewidth=5,fontsize=20,grid=True)
plt.title("Close price of INFY and TCS")
plt.show()


# In[42]:


nse_infy[['4weeks','16weeks','28weeks','40weeks','52weeks']].plot(grid=True,figsize=(20,10),linewidth=5,fontsize=20)
plt.title("Moving Average Of INFY")


# In[43]:


nse_tcs[['4weeks','16weeks','28weeks','40weeks','52weeks']].plot(grid=True,figsize=(20,10),linewidth=5,fontsize=20)
plt.title("Moving Average Of TCS")


# In[44]:


nse_infy[['Close','Volume']].plot(secondary_y=['Volume'],grid=True,figsize=(20,10),linewidth=5,fontsize=20)


# In[45]:


nse_tcs[['Close','Volume']].plot(secondary_y=['Volume'],grid=True,figsize=(20,10),linewidth=5,fontsize=20)


# In[46]:


from pandas.tools.plotting import autocorrelation_plot


# In[47]:


autocorrelation_plot(np.log(nse_infy['Close']))


# In[48]:


autocorrelation_plot(np.log(nse_tcs['Close']))


# In[49]:


nse_infy=nse_infy.drop(['T','PreviousVolume','T1'],axis=1).head()


# In[51]:


nse_tcs=nse_tcs.drop(['T','PreviousVolume','T1'],axis=1).head()


# In[52]:


correlation=nse_infy.corr()['Close']
correlation=pd.DataFrame(abs(correlation))
correlation.Close.sort_values(ascending=False)


# In[54]:


correlation1=nse_tcs.corr()['Close']
correlation1=pd.DataFrame(abs(correlation))
correlation1.Close.sort_values(ascending=False)


# In[55]:


plt.subplots(figsize=(18,18))
sns.heatmap(nse_infy.corr(),annot=True)
plt.title("Correlations Among Features",fontsize = 20)


# In[56]:


plt.subplots(figsize=(18,18))
sns.heatmap(nse_tcs.corr(),annot=True)
plt.title("Correlations Among Features",fontsize = 20)


# In[57]:


x=nse_infy.drop(['Close','Symbol','Series'],axis=1)
y=nse_infy['Close']


# In[58]:


import sklearn.model_selection as model_selection
x_train,x_test,y_train,y_test=model_selection.train_test_split(x,y,test_size=0.20,random_state=200)


# In[59]:


import sklearn.linear_model as linear_model


# In[60]:


reg=linear_model.Ridge(normalize=True,fit_intercept=True)
reg=reg.fit(x_train,y_train)


# In[61]:


greg=model_selection.GridSearchCV(reg,param_grid={'alpha':np.arange(0.1,100,1)})
greg=greg.fit(x_train,y_train)
greg.best_params_


# In[62]:


reg=linear_model.Ridge(normalize=True,fit_intercept=True,alpha=99.1)
reg=reg.fit(x_train,y_train)
reg.coef_


# In[63]:


reg.intercept_


# In[64]:


import sklearn.metrics as metrics
import sklearn.preprocessing as preprocessing


# In[65]:


x_test=preprocessing.normalize(x_test)


# In[66]:


print('MAE:',metrics.mean_squared_error(y_test,greg.predict(x_test)))


# In[67]:


x1=nse_tcs.drop(['Close','Symbol','Series'],axis=1)
y1=nse_tcs['Close']
x1_train,x1_test,y1_train,y1_test=model_selection.train_test_split(x1,y1,test_size=0.20,random_state=200)


# In[68]:


reg1=linear_model.Lasso(max_iter=1000,normalize=True)
reg1=reg1.fit(x1_train,y1_train)


# In[69]:


greg1=model_selection.GridSearchCV(reg1,param_grid={'alpha':np.arange(0.1,100,1).tolist()})
greg1=greg1.fit(x1_train,y1_train)
greg1.best_params_


# In[70]:


reg1=linear_model.Lasso(max_iter=1000,normalize=True,alpha=14.1)
reg1=reg1.fit(x1_train,y1_train)
reg1.coef_


# In[71]:


reg1.intercept_


# In[72]:


x1_test=preprocessing.normalize(x1_test)


# In[73]:


print('MAE:',metrics.mean_squared_error(y1_test,greg1.predict(x1_test)))

