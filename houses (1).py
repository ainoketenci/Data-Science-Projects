#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

X_train = train.drop(columns=['SalePrice'])
y_train = train[['SalePrice']].values.ravel()


# In[2]:


#replace missing observations with mode / mean, drop obs with too many missing values
from pandas.api.types import is_numeric_dtype

def handlemissing(X):    
    null_columns = X.isnull().sum()[X.isnull().sum()>0]
    #train.count() #1460 obs
    null_columns.index.values

    for col in null_columns.index.values: 
        X[col].fillna(X[col].mode()[0], inplace=True)
        
    X.drop(columns=['Id','Alley', 'PoolQC', 'Fence', 'MiscFeature'], inplace=True)
    return X

X_train=handlemissing(X_train)


# In[3]:


#categorical to numeric
from sklearn import preprocessing

def categorical_to_numeric(X):
    obj = X.select_dtypes(include=['object'])
    le = preprocessing.LabelEncoder()
    for i in range(len(obj.columns)):
        le.fit(X[obj.columns[i]])
        X[obj.columns[i]] = le.transform(X[obj.columns[i]])
    return X

X_train=categorical_to_numeric(X_train)


# In[4]:


import matplotlib.pyplot as plt

for col in X_train.columns:
    plt.scatter(X_train[col],y_train)
    plt.xlabel(col) 
    plt.show()


# In[3]:


from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.pipeline import Pipeline


# In[4]:


pipe_linreg = Pipeline([('scaler', StandardScaler()), ('linreg', LinearRegression())])
params=[{'linreg__fit_intercept': [True, False]}]
gs_linreg = GridSearchCV(estimator=pipe_linreg, param_grid=params) 

gs_linreg.fit(X_train, y_train)
print('Best params: %s' % gs_linreg.best_params_)
print('Best training score: %.3f' % gs_linreg.best_score_)


# In[7]:


pipe_ridge = Pipeline([('scaler', StandardScaler()), ('ridge', Ridge())])
params = [{'ridge__alpha': [1.0, 10.0, 100.0]}] 

gs_ridge = GridSearchCV(estimator=pipe_ridge, param_grid=params) 
gs_ridge.fit(X_train, y_train)
print('Best params: %s' % gs_ridge.best_params_)
print('Best training score: %.3f' % gs_ridge.best_score_)


# In[8]:


pipe_lasso = Pipeline([('scaler', StandardScaler()), ('lasso', Lasso())])
params = [{'lasso__alpha': [1.0, 10.0, 100.0]}] 

gs_lasso = GridSearchCV(estimator=pipe_lasso, param_grid=params) 
gs_lasso.fit(X_train, y_train)
print('Best params: %s' % gs_lasso.best_params_)
print('Best training score: %.3f' % gs_lasso.best_score_)




# In[9]:


from sklearn.ensemble import RandomForestRegressor
pipe_randomforest = Pipeline([('scaler', StandardScaler()), ('rforest', RandomForestRegressor())])
params = [{'rforest__n_estimators': [10, 100]}] 

gs_rforest = GridSearchCV(estimator=pipe_randomforest, param_grid=params) 
gs_rforest.fit(X_train, y_train)
print('Best params: %s' % gs_rforest.best_params_)
print('Best training score: %.3f' % gs_rforest.best_score_)


# In[10]:


X_test = test
ids = X_test['Id']
X_test=handlemissing(X_test)
X_test=categorical_to_numeric(X_test)


# In[21]:


from sklearn.model_selection import train_test_split


X_train1, X_val, y_train1,y_val = train_test_split(X_train, y_train)

randomf = RandomForestRegressor(n_estimators=100)
randomf.fit(X_train1, y_train1)
print(randomf.score(X_val, y_val))

scaler = StandardScaler()
scaler.fit(X_test)
scaler.transform(X_test)

y_pred = randomf.predict(X_test)
y_pred


# In[22]:


result = pd.DataFrame({'Id':ids, 'SalePrice':y_pred})


# In[23]:


result.to_csv(r'results.csv', index=False)


# In[24]:


#First preprocessed (handled missing values, converted to numeric and scaled) and then fit Linear regression,
# Lasso, Ridge and Random forest. By cross validation obtained more than 80% r2 scores in each, except in 
#Linear regression (seemed to have some problem). 
# Chose Random Forest regressor with highest 87% r2 score, and when submitting to Kaggle obtained 15% score.


# In[ ]:




