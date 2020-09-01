#!/usr/bin/env python
# coding: utf-8

# In[1]:


## Importing Libraries
import pandas as pd
import numpy as np
from sklearn import preprocessing
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier as rfc
from sklearn.neighbors import KNeighborsClassifier


# In[2]:


##Reading file
df1 = pd.read_csv('C:/Users/TOTAGOUSER4/Documents/Totago Technologies/David/Data Science/Projects/DSN Expresso Churn Prediction/Train.csv')
df2 = pd.read_csv('C:/Users/TOTAGOUSER4/Documents/Totago Technologies/David/Data Science/Projects/DSN Expresso Churn Prediction/Test.csv')


# In[3]:


##Checkpoint
data1 = df1.copy()
data2 = df2.copy()


# In[4]:


##To display entire dataset
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows',  None)
df1.head()


# In[ ]:





# In[5]:


df1.info()


# In[5]:


df1['FREQ_TOP_PACK'].unique()


# In[49]:


df1['FREQ_TOP_PACK'].plot()


# In[6]:


## Fill null Tenure with mode (K>24)
df1['REGION'].fillna('Unknown', inplace=True)
df2['REGION'].fillna('Unknown', inplace=True)


# In[7]:


##Fill null Montant with the mean.
df1['MONTANT'].fillna(np.mean(df1['MONTANT']), inplace=True)
df2['MONTANT'].fillna(np.mean(df2['MONTANT']), inplace=True)


# In[8]:


##Fill null Frequency_Rech with the mean.
df1['FREQUENCE_RECH'].fillna(np.mean(df1['FREQUENCE_RECH']), inplace=True)
df2['FREQUENCE_RECH'].fillna(np.mean(df2['FREQUENCE_RECH']), inplace=True)


# In[9]:


##Fill null REVENUE with the mean.
df1['REVENUE'].fillna(np.mean(df1['REVENUE']), inplace=True)
df2['REVENUE'].fillna(np.mean(df2['REVENUE']), inplace=True)


# In[10]:


##Fill null ARPU_SEGMENT with the mean.
df1['ARPU_SEGMENT'].fillna(np.mean(df1['ARPU_SEGMENT']), inplace=True)
df2['ARPU_SEGMENT'].fillna(np.mean(df2['ARPU_SEGMENT']), inplace=True)


# In[11]:


##Fill null FREQUENCE with the mean.
df1['FREQUENCE'].fillna(np.mean(df1['FREQUENCE']), inplace=True)
df2['FREQUENCE'].fillna(np.mean(df2['FREQUENCE']), inplace=True)


# In[12]:


##Fill null DATA_VOLUME with the mean.
df1['DATA_VOLUME'].fillna(np.mean(df1['DATA_VOLUME']), inplace=True)
df2['DATA_VOLUME'].fillna(np.mean(df2['DATA_VOLUME']), inplace=True)


# In[13]:


##Fill null ON_NET with the mean.
df1['ON_NET'].fillna(np.mean(df1['ON_NET']), inplace=True)
df2['ON_NET'].fillna(np.mean(df2['ON_NET']), inplace=True)


# In[14]:


##Fill null ORANGE with the mean.
df1['ORANGE'].fillna(np.mean(df1['ORANGE']), inplace=True)
df2['ORANGE'].fillna(np.mean(df2['ORANGE']), inplace=True)


# In[ ]:





# In[52]:


df1.info()


# In[15]:


df1.columns


# In[16]:


y = df1[['CHURN']]
         
X1 = df1[['REGION', 'TENURE', 'MONTANT', 'FREQUENCE_RECH', 'REVENUE',
       'ARPU_SEGMENT', 'FREQUENCE', 'DATA_VOLUME', 'MRG', 'REGULARITY']]

X2 = df2[['REGION', 'TENURE', 'MONTANT', 'FREQUENCE_RECH', 'REVENUE',
       'ARPU_SEGMENT', 'FREQUENCE', 'DATA_VOLUME', 'MRG', 'REGULARITY']]


# In[17]:


##Dummies
X1 = pd.get_dummies(X1)
X2 = pd.get_dummies(X2)


# In[18]:


scaler = StandardScaler()
scaler.fit(X1)

X1 = scaler.transform(X1)
X2 = scaler.transform(X2)


# In[19]:


## Logistic Regression
Reg = LogisticRegression()


# In[20]:


##Fitting the data
Reg.fit(X1, y)


# In[21]:


##Accuracy of model
Reg.score(X1, y)


# In[30]:


##Using K Nearest Neighbors
classifier = KNeighborsClassifier(n_neighbors=3)


# In[31]:


classifier.fit(X1, y)


# In[ ]:


##Accuracy
classifier.score(X1, y)


# In[ ]:


y4 = classifier.predict(X2)


# In[25]:


##Using random forest
rfc = rfc()


# In[26]:


rfc.fit(X1, y)


# In[27]:


##Accuracy
rfc.score(X1, y)


# In[22]:


y2 = rfc.predict(X2)


# In[19]:





# In[21]:





# In[22]:





# In[23]:





# In[24]:





# In[26]:





# In[28]:





# In[29]:


y3


# In[30]:


##Saving the prediction
z = pd.read_csv('C:/Users/TOTAGOUSER4/Documents/Totago Technologies/David/Data Science/Projects/DSN Expresso Churn Prediction/sample_submission.csv')

output = pd.DataFrame({'User_ID':z['user_id'], 'Churn': y3})
output.to_csv('C:/Users/TOTAGOUSER4/Documents/Totago Technologies/David/Data Science/Projects/DSN Expresso Churn Prediction/my_prediction3.csv', index = False)


# In[ ]:




