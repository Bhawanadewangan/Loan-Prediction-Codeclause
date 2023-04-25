#!/usr/bin/env python
# coding: utf-8

# ### Loan Prediction -LOANS are the major requirement of the modern world. By this only, Banks get a major part of the total profit. It is beneficial for students to manage their education and living expenses, and for people to buy any kind of luxury like houses, cars, etc.But when it comes to deciding whether the applicant’s profile is relevant to be granted with loan or not. Banks have to look after many aspects.So, here we will be using Machine Learning with Python to ease their work and predict whether the candidate’s profile is relevant or not using key features like Marital Status, Education, Applicant Income, Credit History, etc.
# 

# # Loan Prediction using Machine Learning

# In[1]:


# import some libraries 


# In[3]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


#  #Import dataset using pandas

# In[7]:


data_set=pd.read_csv("train_u6lujuX_CVtuZ9i.csv")


# ## Data Preprocessing and Visualization

# In[8]:


data_set.head()


# In[9]:


data_set.shape


# In[10]:


data_set.info()


# In[11]:


data_set.describe()


# In[12]:


data_set.isnull().sum()


# In[13]:


data_set.drop(["Loan_ID","Dependents"],axis=1,inplace = True)


# In[14]:


data_set


# In[15]:


## Dealing with null value (categorical value)
cols = data_set[["Gender","Married","Self_Employed"]]
for i in cols:
    data_set[i].fillna(data_set[i].mode().iloc[0], inplace=True)


# In[16]:


data_set.isnull().sum()


# In[17]:


## Dealing with Numerical Values missing_data
num_cols=data_set[["LoanAmount","Loan_Amount_Term","Credit_History"]]
for i in num_cols:
    data_set[i].fillna(data_set[i].mean(axis=0), inplace=True)


# In[18]:


data_set.isnull().sum()


# In[19]:


## Visualizations
def bar_chart(col):
    Approved=data_set[data_set["Loan_Status"]=="Y"][col].value_counts()
    Disapproved = data_set[data_set["Loan_Status"]=="N"][col].value_counts()
    
    df1 = pd.DataFrame([Approved,Disapproved])
    df1.index=["Approved","Disapproved"]
    df1.plot(kind="bar")
   


# In[20]:


bar_chart("Gender")


# In[21]:


bar_chart("Married")


# In[22]:


bar_chart("Education")


# In[23]:


bar_chart("Self_Employed")


# In[24]:


from sklearn.preprocessing import OrdinalEncoder
ord_enc = OrdinalEncoder()
data_set[['Gender','Married','Education','Self_Employed','Property_Area','Loan_Status']]=ord_enc.fit_transform(data_set[['Gender','Married','Education','Self_Employed','Property_Area','Loan_Status']])


# In[25]:


data_set.head()


# In[26]:


# change in int 
data_set[['Gender','Married','Education','Self_Employed','Property_Area','Loan_Status']].astype('int')


# In[27]:


test_data = pd.read_csv("test_Y3wMUE5_7gLdaTN.csv")


# In[28]:


test_data.head()


# In[29]:


test_data.shape


# In[30]:


test_data.info()


# In[31]:


test_data.describe()


# In[32]:


test_data.isnull().sum()


# In[33]:


test_data.drop(["Loan_ID","Dependents"],axis=1,inplace = True)


# In[34]:


test_data


# In[36]:


col_test=["Gender","Self_Employed"]
for i in col_test:
    test_data[i].fillna(test_data[i].mode().iloc[0],inplace=True)


# In[37]:


test_data.isnull().sum()


# In[38]:


cols_tests=["LoanAmount","Loan_Amount_Term","Credit_History"]
for j in cols_tests:
    test_data[j].fillna(test_data[j].mean(axis=0),inplace=True)


# In[39]:


test_data.isnull().sum()


# ## Splitting Datasets

# In[40]:


from sklearn.preprocessing import OrdinalEncoder
ord_enc_test = OrdinalEncoder()
test_data[['Gender','Married','Education','Self_Employed','Property_Area']]=ord_enc_test.fit_transform(test_data[['Gender','Married','Education','Self_Employed','Property_Area']])


# In[41]:


test_data


# In[42]:


test_data[['Gender','Married','Education','Self_Employed','Property_Area']].astype('int')


# In[45]:


#split the data 
from sklearn.model_selection import train_test_split
x = data_set.drop("Loan_Status",axis=1)
y = data_set["Loan_Status"]

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2,random_state=0)


# In[46]:


print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)


# ### Model Training and Evaluation

# #GaussianNB 
# #GridSearchCV 
# #svc #XGBClassifier 
# #DecisionTreeClassifier 
# #RandomizedSearchCV 
# #RandomForestClassifier
# 
# 

# In[47]:


from sklearn.naive_bayes import GaussianNB

gfc = GaussianNB()
gfc.fit(x_train, y_train)
pred1 = gfc.predict(x_test)


# In[48]:


pret_test=gfc.predict(test_data)
pret_test


# In[49]:


from sklearn.metrics import precision_score, recall_score,accuracy_score
def loss(y_true, y_pred):
    prec_score= precision_score(y_true, y_pred)
    rec_score = recall_score(y_true, y_pred)
    acc_score = accuracy_score(y_true, y_pred)
    
    print(prec_score)
    print(rec_score)
    print(acc_score)


# In[50]:


loss(y_test, pred1)


# In[51]:


from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV

svc =SVC()
#defining parameter range
param_grid = {'C' : [0.1, 1, 10, 100, 1000],
              "gamma" : [1, 0.1, 0.01, 0.001,0.0001],
              "kernel" : ['rbf']}
grid = GridSearchCV(SVC(),param_grid, refit=True, verbose = 3)
grid.fit(x_train, y_train)


# In[52]:


grid.best_params_


# In[53]:


svc = SVC(C=0.1, gamma= 1, kernel='rbf')
svc.fit(x_train, y_train)
pred2 = svc.predict(x_test)
loss(y_test,pred2)


# In[54]:


from xgboost import XGBClassifier

xgb = XGBClassifier(learning_rate = 0.1,
n_estimators = 1000,
max_depth = 3,
min_child_weight = 1,
gamma = 0,
subsample = 0.8,
colsample_bytree = 0.8, 
objective = 'binary:logistic',
nthread = 4, 
scale_pos_weight = 1,
seed= 27)
xgb.fit(x_train, y_train)
pred3 = xgb.predict(x_test)
loss(y_test, pred3)


# In[55]:


from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import RandomizedSearchCV

def randomized_search(params, runs=20, clf=DecisionTreeClassifier(random_state=2)):
    rand_clf = RandomizedSearchCV(clf, params, n_iter=runs, cv=5, n_jobs=-1, random_state=2)
    rand_clf.fit(x_train, y_train)
    best_model = rand_clf.best_estimator_
     
    #Extract best score
    best_score = rand_clf.best_score_
    
    # print best score 
    print("Training score:{: .3f}".format(best_score))
    
    #predict test set labels
    y_pred = best_model.predict(x_test)
    
    # compute accuracy
    accuracy = accuracy_score(y_test, y_pred)
    
    # print accuracy
    print("test_score: {: .3f}".format(accuracy))
    
    return best_model


# In[56]:


randomized_search(params={'criterion':['entropy','gini'],'splitter':['random','best'],
                         'min_weight_fraction_leaf':[0.0, 0.0025, 0.005, 0.0075, 0.01],
                         'min_samples_split':[2, 3, 4, 5, 6, 8, 10],
                         'min_samples_leaf':[1, 0.01, 0.02, 0.03, 0.04],
                         'min_impurity_decrease':[0.0, 0.0005, 0.005, 0.05, 0.10, 0.15, 0.2],
                         'max_leaf_nodes':[10, 15, 20, 25, 30, 35, 40, 45, 50, None],
                         'max_features':['auto', 0.95, 0.90, 0.85, 0.80, 0.75, 0.70],
                         'max_depth':[None, 2, 4, 6, 8],
                         'min_weight_fraction_leaf':[0.0, 0.0025, 0.005, 0.0075, 0.01, 0.05]     
                                                    })


# In[57]:


ds=DecisionTreeClassifier(max_depth=8, max_features=0.9, max_leaf_nodes=30,
                       min_impurity_decrease=0.05, min_samples_leaf=0.02,
                       min_samples_split=10, min_weight_fraction_leaf=0.005,
                       random_state=2, splitter='random')
ds.fit(x_train, y_train)
pred4 = ds.predict(x_test)
loss(y_test, pred4)


# In[58]:


pret_test_ds=ds.predict(test_data)
pret_test_ds


# In[59]:


from sklearn.ensemble import RandomForestClassifier
randomized_search(params={'min_samples_leaf':[1,2,4,6,8,10,20,30],
                         'min_impurity_decrease':[0.0, 0.01, 0.05,0.10, 0.15, 0.2],
                         'max_features':['auto', 0.8, 0.7, 0.6, 0.5, 0.4],
                         'max_depth':[None, 2, 4, 6, 8, 10, 20],
                         }, clf= RandomForestClassifier(random_state=2))


# In[60]:


RFC=RandomForestClassifier(max_depth=2, max_features=0.5,
                       min_impurity_decrease=0.01, min_samples_leaf=10,
                       random_state=2)
RFC.fit(x_train, y_train)
pred5 = RFC.predict(x_test)
loss(y_test, pred5)


# In[61]:


pret_test_RFC=RFC.predict(test_data)
pret_test_RFC


# In[62]:


import joblib
joblib.dump(ds, "model_ds.ds")
model = joblib.load('model_ds.ds')
model.predict(x_test)


# In[63]:


#Conclusion : Random Forest Classifier or Decision Tree Classifier is giving the best accuracy with an accuracy score of 82% 
#for the testing dataset.


# In[64]:


df=pd.DataFrame(test_data)


# In[65]:


df["Loan_Status_pred"]=pret_test_ds


# In[66]:


df


# In[ ]:




