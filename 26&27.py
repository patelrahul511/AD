#!/usr/bin/env python
# coding: utf-8

# In[26]:


from sklearn.cluster import DBSCAN
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt


# In[27]:


df = pd.read_csv(r"C:\Users\Rama_krishna\OneDrive\Desktop\Wholesale customers data.csv")
df.head()


# In[28]:


df.Channel.unique()


# In[29]:


import seaborn as sns 
sns.countplot(x='Channel',data=df)
plt.show()


# In[30]:


sns.countplot(x=df['Region'])
plt.show()


# In[31]:


import warnings
warnings.filterwarnings('ignore')
sns.distplot(x=df['Milk'])
plt.show()


# In[32]:


sns.distplot(x=df['Fresh'])
plt.show()


# In[33]:


sns.distplot(x=df['Grocery'])
plt.show()


# In[34]:


sns.distplot(x=df['Detergents_Paper'])
plt.show()


# In[35]:


sns.distplot(x=df['Delicassen'])
plt.show()


# In[36]:


df.info()


# In[37]:


from sklearn.preprocessing import StandardScaler
stscaler = StandardScaler()
X=stscaler.fit_transform(df)


# In[38]:


X


# In[39]:


import scipy.cluster.hierarchy as sch


# In[40]:


plt.figure(figsize=(20,6))
dendo = sch.dendrogram(sch.linkage(X,method='ward'))
plt.title('Dendogram')
plt.xlabel('Customer data')
plt.ylabel('Eucl Distance ')
plt.show()


# In[41]:


len(set(dendo['color_list']))


# In[42]:


from sklearn.cluster import AgglomerativeClustering


# In[43]:


model = AgglomerativeClustering(n_clusters=3)
cluster=model.fit_predict(X)


# In[44]:


cluster


# In[45]:


df


# In[46]:


group_num=pd.DataFrame(cluster,columns=['Group'])
group_num


# In[47]:


cust_group_data=pd.concat([df,group_num],axis=1)
cust_group_data


# In[ ]:





# # Kmeans

# In[48]:


from sklearn.cluster import KMeans
wcss=[]
for i in range(2,11):
    Kmeans=KMeans(n_clusters=i,init='k-means++',random_state=42)
    Kmeans.fit(X)
    wcss.append(Kmeans.inertia_)


# In[49]:


wcss


# In[50]:


plt.plot(range(2,11),wcss)
plt.title('The Elbow Method')
plt.xlabel('number of clusters')
plt.ylabel('wcss')
plt.show()


# # DBSCAN

# In[52]:


from sklearn.cluster import DBSCAN


# In[57]:


dbscan = DBSCAN(eps=3.2,min_samples=10)
dbscan.fit(X)


# 

# In[60]:


dbscan.labels_


# In[62]:


cl=pd.DataFrame(dbscan.labels_,columns=['cluster'])
df_cluster=pd.concat([df,cl],axis=1)


# In[63]:


df_cluster


# In[ ]:




