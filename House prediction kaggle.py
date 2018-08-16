#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 14 17:57:13 2018

@author: vikramreddy
"""
import pandas as pd
import numpy as np
#reading data
train=pd.read_csv('train.csv')
test=pd.read_csv('test.csv')
#after observing the data description I am removing some unimportant columns according to me
remov_col=['TotalBsmtSF','TotRmsAbvGrd','GarageQual','OverallQual','LandContour','LotFrontage','Alley',
           'HalfBath','FullBath','BsmtHalfBath','BsmtFullBath']
train=train.drop(remov_col,axis=1)
test=test.drop(remov_col,axis=1)
#checking for null value count and removing those columns which has more than1000 null values
#no point in imputing them it does not add any varince to the model
a=pd.DataFrame({'col':list(train),'val':train.isnull().sum(axis=0)})
for i in range(70):
    if(a.iloc[i,1]>1000):
        a.iloc[i,1]=np.NAN
    else:''
        
a=a.dropna(how='any')
train=train[a.iloc[:,0]]

#check for null values
train.isnull().any().any()
test.isnull().any()
#checking for unique value
for i in range(len(list(train))):
    if(train.iloc[:,i].nunique()==1):
        print(i)
        train.iloc[:,i]=np.NaN
    else:''
    #numerical columns  in train and test to impute with median and mean
num_col=list(train.describe())
num_col_test=list(test.describe())
for i in range(len(num_col)):
    train[num_col[i]]=train[num_col[i]].fillna(train[num_col[i]].median(),axis=0)
for i in range(len(num_col_test)):
    test[num_col_test[i]]=test[num_col_test[i]].fillna(test[num_col_test[i]].median(),axis=0)
    #dropping duplicate columns
train=train.drop_duplicates(keep=False)
test=test.drop_duplicates(keep=False)
#factor columns of train and test and replacing with mode values
fact_col=list(train.drop(num_col,axis=1))
fact_col_test=list(test.drop(num_col_test,axis=1))
for i in range(len(fact_col)):
        train[fact_col[i]]=train[fact_col[i]].astype('category')
    
for i in range(len(fact_col_test)):
        test[fact_col_test[i]]=test[fact_col_test[i]].astype('category')
#numerical factors
for i in range(len(fact_col)):
    train[fact_col[i]]=pd.factorize(train[fact_col[i]])[0]
for i in range(len(fact_col_test)):
    test[fact_col[i]]=pd.factorize(test[fact_col[i]])[0]
#imputing factor columns with mode   
for i in range(len(fact_col)):
    train[fact_col[i]]=train[fact_col[i]].fillna(train[fact_col[i]].mode(),axis=0)
for i in range(len(num_col_test)):
    test[fact_col_test[i]]=test[fact_col_test[i]].fillna(test[fact_col_test[i]].mode(),axis=0)
#converting those back to factor levels
for i in range(len(fact_col)):
        train[fact_col[i]]=train[fact_col[i]].astype('category')

for i in range(len(fact_col_test)):
        test[fact_col_test[i]]=test[fact_col_test[i]].astype('category')
#converting factor column to dummy columns
train=pd.get_dummies(train)
test=pd.get_dummies(test)
#finding correlation matrix
corr_matrix=train.corr()
# Select upper triangle of correlation matrix
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))

# Find index of feature columns with correlation greater than 0.95
to_drop = [column for column in upper.columns if any(upper[column] > 0.95)]
train=train.drop(to_drop,axis=1)
from sklearn.preprocessing import MinMaxScaler
#scaling varibels
scaler=MinMaxScaler()
#training and test data
X_Train=train.drop(['Id','SalePrice'],axis=1)
X_Train=scaler.fit_transform(X_Train)
y_Train=np.log(train.SalePrice)
X_Test=scaler.fit_transform(test.drop(['Id'],axis=1))

from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

X_train, X_test, y_train, y_test = train_test_split( X_Train, y_Train, test_size=0.20, random_state=42)
######lasso regression

from sklearn.metrics import mean_squared_error


ridge_model=Ridge(alpha=0.01,fit_intercept=True, max_iter=10000,
   normalize=True)
ridge_model.fit(X_train,y_train)
ridge_predictions=ridge_model.predict(X_test)
ridge_model.score(X_test,y_test)  ####0.8972859055514335

r2_score(ridge_predictions, y_test)###0.87362609927304136
np.sqrt(mean_squared_error(y_test, ridge_predictions)) #0.13

#print(ridge_model.score(X_test, y_test))
from sklearn.ensemble import  RandomForestRegressor

rf_model=RandomForestRegressor(n_estimators=50,criterion='mse',max_features='auto',min_samples_split=10)
rf_model.fit(X_train,y_train)
rf_model.score(X_test,y_test)#0.87257102983488932
rf_model.score(X_train,y_train)#0.96222355085277811
rf_pred=rf_model.predict(X_test)
np.sqrt(mean_squared_error(y_test,rf_pred)) #0.15



#################
#
# ridge model is performing well and Random forest is overfitting
#
################
from math import e

#### fitting on total train data
ridge_model.fit(X_Train,y_Train)
# predicting on test data for submission
ridge_pred=e**(ridge_model.predict(X_Test))
submission={'Id':test.Id,'SalePrice':ridge_pred}
submission=pd.DataFrame(submission)
submission.to_csv('submission.csv', index=False, header=True)
sub=pd.read_csv('submission.csv')
###############
#
#
# RMSLE of 1.8 using this predictions(after submitting) lets optimise this using PCA
#
######################



#########trying with PCA

######################
from sklearn.decomposition import PCA
#importing again as some factors are not available in test data
#reading data
train_pca=pd.read_csv('train.csv')
test_pca=pd.read_csv('test.csv')
total=pd.concat([train_pca.drop(['Id','SalePrice'],axis=1),test_pca.drop(['Id'],axis=1)])
remov_col=['TotalBsmtSF','TotRmsAbvGrd','GarageQual','OverallQual','LandContour','LotFrontage','Alley',
           'HalfBath','FullBath','BsmtHalfBath','BsmtFullBath']

total=total.drop(remov_col,axis=1)

#imputing null values in numerical and factor columns saperately
num_col=list(total.describe())
for i in range(len(num_col)):
    total[num_col[i]]=total[num_col[i]].fillna(total[num_col[i]].median(),axis=0)
fact_col=list(total.drop(num_col,axis=1))
for i in range(len(fact_col)):
        total[fact_col[i]]=total[fact_col[i]].astype('category')

#numerical factors
for i in range(len(fact_col)):
    total[fact_col[i]]=pd.factorize(total[fact_col[i]])[0]

#imputing with mode in factor columns
for i in range(len(fact_col)):
    total[fact_col[i]]=total[fact_col[i]].fillna(total[fact_col[i]].mode(),axis=0)
    
#converting those back to factor levels
for i in range(len(fact_col)):
        total[fact_col[i]]=total[fact_col[i]].astype('category')
total=pd.get_dummies(total)

pca=PCA(n_components=311)
scaled_total=pd.DataFrame(scaler.fit_transform(total),columns=list(total))
pca.fit_transform(scaled_total)

var= pca.explained_variance_ratio_
var_cum=np.cumsum(np.round(var, decimals=4)*100)
import matplotlib.pyplot as plt
plt.plot(var_cum)
pca.new = PCA(n_components=170)
pca.new.fit(scaled_total)

pca_feat=pd.DataFrame(pca.fit_transform(scaled_total))
pca_train=pca_feat.iloc[0:1460,:170]
pca_test=pca_feat.iloc[1460:2920,:170]
X_train_p, X_test_p, y_train_p, y_test_p = train_test_split( pca_train, y_Train, test_size=0.20, random_state=42)

ridge_model.fit(X_train_p,y_train_p)
ridge_predictions_PCA=ridge_model.predict(X_test_p)
ridge_model.score(X_test_p,y_test_p)  ####0.89

r2_score(ridge_predictions_PCA, y_test_p)###0.8705
np.sqrt(mean_squared_error(y_test_p, ridge_predictions_PCA)) #0.14

#rf model
rf_model.fit(X_train_p,y_train_p)
rf_model.score(X_test_p,y_test_p)#0.7644
rf_model.score(X_train_p,y_train_p)#0.94222355085277811
rf_pred=rf_model.predict(X_test_p)
np.sqrt(mean_squared_error(y_test_p,rf_pred)) #0.21

#Predictions on ridge model
########                                            ################
#fitting on whole data on PCA ridge predictions  RMSLE changed from 1.8 to 0.8
#                                               ###################
#######
ridge_model.fit(pca_train,y_Train)
predictions_p=e**(ridge_model.predict(pca_test))
submission_p={'Id':test.Id,'SalePrice':predictions_p}
submission_p=pd.DataFrame(submission_p)
submission_p.to_csv('pca_pred_new.csv',index=False, header=True)

##############
#
#
# with out removing any column from data
#
#################
train_pca=pd.read_csv('train.csv')
test_pca=pd.read_csv('test.csv')
total=pd.concat([train_pca.drop(['Id','SalePrice'],axis=1),test_pca.drop(['Id'],axis=1)])
a=pd.DataFrame({'col':list(total),'val':total.isnull().sum(axis=0)})
for i in range(len(list(total))):
    if(a.iloc[i,1]>2000):
        a.iloc[i,1]=np.NAN
    else:''
        
a=a.dropna(how='any')
total=total[a.iloc[:,0]]

#imputing null values in numerical and factor columns saperately
num_col=list(total.describe())
for i in range(len(num_col)):
    total[num_col[i]]=total[num_col[i]].fillna(total[num_col[i]].median(),axis=0)
fact_col=list(total.drop(num_col,axis=1))
for i in range(len(fact_col)):
        total[fact_col[i]]=total[fact_col[i]].astype('category')

#numerical factors
for i in range(len(fact_col)):
    total[fact_col[i]]=pd.factorize(total[fact_col[i]])[0]

#imputing with mode in factor columns
for i in range(len(fact_col)):
    total[fact_col[i]]=total[fact_col[i]].fillna(total[fact_col[i]].mode(),axis=0)
    
#converting those back to factor levels
for i in range(len(fact_col)):
        total[fact_col[i]]=total[fact_col[i]].astype('category')
total=pd.get_dummies(total)

pca=PCA(n_components=294)
scaled_total=pd.DataFrame(scaler.fit_transform(total),columns=list(total))
pca.fit_transform(scaled_total)

var= pca.explained_variance_ratio_
var_cum=np.cumsum(np.round(var, decimals=4)*100)
import matplotlib.pyplot as plt
plt.plot(var_cum)
pca.new = PCA(n_components=170)
pca.new.fit(scaled_total)

pca_feat=pd.DataFrame(pca.fit_transform(scaled_total))
pca_train=pca_feat.iloc[0:1460,:170]
pca_test=pca_feat.iloc[1460:2920,:170]
X_train_p, X_test_p, y_train_p, y_test_p = train_test_split( pca_train, y_Train, test_size=0.20, random_state=42)

ridge_model.fit(X_train_p,y_train_p)
ridge_predictions_PCA=ridge_model.predict(X_test_p)
ridge_model.score(X_test_p,y_test_p)  ####0.89

r2_score(ridge_predictions_PCA, y_test_p)###0.8705
np.sqrt(mean_squared_error(y_test_p, ridge_predictions_PCA)) #0.14

#rf model
rf_model.fit(X_train_p,y_train_p)
rf_model.score(X_test_p,y_test_p)#0.7644
rf_model.score(X_train_p,y_train_p)#0.94222355085277811
rf_pred=rf_model.predict(X_test_p)
np.sqrt(mean_squared_error(y_test_p,rf_pred)) #0.21

#Predictions on ridge model
########                                           ################### 
#fitting on whole data on PCA ridge predictions RMSLE changed to 0.14021
#                                               ######################
#######
ridge_model.fit(pca_train,y_Train)
predictions_p_new=e**(ridge_model.predict(pca_test))
submission_p_new={'Id':test.Id,'SalePrice':predictions_p_new}
submission_p_new=pd.DataFrame(submission_p_new)
submission_p_new.to_csv('pca_pred_new11.csv',index=False, header=True)
submission.to_csv('submission.csv', index=False, header=True)