#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


df = pd.read_csv('heart .csv')


# In[3]:


df.head()


# In[4]:


df.columns


# In[39]:


df.shape


# In[5]:


df = df.rename(columns={'trestbps':'bp','thalach':'rhb'})
df.head()


# In[6]:


corr_matrix = df.corr()
corr_matrix


# In[7]:


plt.figure(figsize =[10,10])
heatmap = sns.heatmap(corr_matrix,annot=True,square=True)
heatmap


# In[8]:


from sklearn.preprocessing import MinMaxScaler
normalize = MinMaxScaler()
x = df.drop('target',axis = True)


# In[9]:


normalize.fit(x)


# In[10]:


new_normalize = normalize.transform(x)
new_normalize


# In[11]:


norm = pd.DataFrame(new_normalize)


# In[12]:


norm.head()


# In[15]:


x = norm
y = df[['target']]


# In[49]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=0)


# In[50]:


from sklearn import svm
clf = svm.SVC(kernel='rbf')


# In[51]:


clf.fit(x_train,y_train)


# In[52]:


clf.predict(x_test)


# In[53]:


x_test.head()


# In[54]:


clf.score(x_test,y_test)


# In[55]:


y_pred = clf.predict(x_test)


# In[57]:


from sklearn.metrics import classification_report,confusion_matrix,precision_score,recall_score
cr = classification_report(y_pred,y_test)
cr
cl = confusion_matrix(y_pred,y_test)
cl


# In[ ]:




