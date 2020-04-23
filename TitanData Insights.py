
# coding: utf-8

# In[28]:


import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt


# In[29]:


data=pd.read_csv('train.csv')
data.shape


# In[30]:


data.head()


# In[31]:


data.isnull().sum()


# In[32]:


data["Age"].fillna(data["Age"].mean(),inplace=True)


# In[33]:


temp=data.pivot_table(values="PassengerId",columns="Survived",index="Sex",aggfunc="count")


# In[34]:


temp.plot.bar()


# In[35]:


temp


# In[36]:


temp.loc[:,1].plot.bar()


# In[37]:


data1=data[data["Survived"]==1]


# In[38]:


temps=data1.groupby(["Age","Sex"])["Survived"].count()


# In[39]:


fig=plt.figure(figsize=(25,7))
ax=fig.add_axes()
ax=temps.plot.bar()
ax.set_ylabel("Survived",size=20)


# In[40]:


data1=data[data["Survived"]==1]
data2=data[data["Survived"]==0]


# In[41]:


plt.scatter(data1["Age"],data1["Fare"],label="Survived")
plt.scatter(data2["Age"],data2["Fare"],color='r',label="Not Survived",alpha=.5)
plt.legend()

