#!/usr/bin/env python
# coding: utf-8

# In[1]:


from pandas import read_csv
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier


# In[3]:


name =['preg','plas','pres','skin','test','mass','pedi','age','class']
dataframe = read_csv(r"C:\Users\Rama_krishna\Downloads\pima-indians-diabetes.data.csv",names=name)
dataframe.head()


# In[4]:


array=dataframe.values
X=array[:,0:8]
Y=array[:,8]


# In[5]:


num_trees =100
seed = 7
k_fold = KFold(n_splits=10,shuffle=True,random_state=seed)
model = RandomForestClassifier(n_estimators=num_trees,max_features=3)
results = cross_val_score(model,X,Y,cv=k_fold)
k_fold


# In[6]:


result = cross_val_score(model,X,Y,cv=k_fold)
print(result.mean())


# In[7]:


### import required libraries
from pandas import read_csv
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier


# In[8]:


name =['preg','plas','pres','skin','test','mass','pedi','age','class']
dataframe = read_csv(r"C:\Users\Rama_krishna\Downloads\pima-indians-diabetes.data.csv",names=name)
dataframe.head()


# In[9]:


array=dataframe.values
X=array[:,0:8]
Y=array[:,8]


# In[10]:


## set the parameters
seed = 7
k_fold = KFold(n_splits=10,shuffle=True,random_state=seed)
cart = DecisionTreeClassifier()
num_trees = 100
model = BaggingClassifier(estimator=cart,n_estimators=num_trees,random_state=seed)


# In[11]:


results = cross_val_score(model,X,Y,cv=k_fold)
print(results.mean())


# In[12]:


## ADA boost classifier
from sklearn.ensemble import AdaBoostClassifier
name =['preg','plas','pres','skin','test','mass','pedi','age','class']
dataframe = read_csv(r"C:\Users\Rama_krishna\Downloads\pima-indians-diabetes.data.csv",names=name)
print(dataframe.head())
array=dataframe.values
X=array[:,0:8]
Y=array[:,8]


# In[13]:


num_trees =10
seed = 7
k_fold = KFold(n_splits=10,shuffle=True,random_state=seed)
model = AdaBoostClassifier(n_estimators=num_trees,random_state=seed)
results = cross_val_score(model,X,Y,cv=k_fold)
print(results.mean())


# In[14]:


from pandas import read_csv
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import VotingClassifier


# In[15]:


name =['preg','plas','pres','skin','test','mass','pedi','age','class']
dataframe = read_csv(r"C:\Users\Rama_krishna\Downloads\pima-indians-diabetes.data.csv",names=name)
dataframe.head()


# In[17]:


array=dataframe.values
X=array[:,0:8]
Y=array[:,8]
kfold = KFold(n_splits=10,shuffle=True,random_state=7)


# In[19]:


estimators=[]
model1 = LogisticRegression(max_iter=500)
estimators.append(('logistic',model1))
model2 = DecisionTreeClassifier()
estimators.append(('cart',model2))
model3 = SVC()
estimators.append(('svm',model3))


# In[20]:


ensemble = VotingClassifier(estimators)
results = cross_val_score(ensemble, X, Y, cv=kfold)
print(results.mean())


# In[ ]:




