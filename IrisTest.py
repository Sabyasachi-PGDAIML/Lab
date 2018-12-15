# -*- coding: utf-8 -*-
"""
Created on Sat Dec 15 17:32:06 2018

@author: Sabya
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from scipy.stats import zscore
from sklearn.metrics import accuracy_score, confusion_matrix, recall_score, precision_score, auc
colnames = ['sepal length','sepal width','petal length','petal width','class']
iris_df = pd.read_csv('iris.data',names= colnames)
iris_df["sepal length"][iris_df["sepal length"] == 0] = iris_df["sepal length"].median()
iris_df["sepal width"][iris_df["sepal width"] == 0] = iris_df["sepal width"].median()
iris_df["petal length"][iris_df["petal length"] == 0] = iris_df["petal length"].median()
iris_df["petal width"][iris_df["petal width"] == 0] = iris_df["petal width"].median()
iris_df["class"][iris_df["class"]=="Iris-setosa"] = 0
iris_df["class"][iris_df["class"]=="Iris-versicolor"] = 1
iris_df["class"][iris_df["class"]=="Iris-virginica"] = 2
iris_df["class"] = iris_df["class"].astype(np.int32)
iris_df.drop('sepal width',axis = 1,inplace=True)
X = iris_df.drop('class',axis=1)
Y = iris_df['class']
X = X.drop('petal length',axis=1)
X = X.apply(zscore)
test_size = 0.20
seed = 7 
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size, random_state=seed)
def y2indicator(y, K):
    N = len(y)
    ind = np.zeros((N, K))
    for i in range(N):
        ind[i, y[i]] = 1
    return ind
K = len(set(Y_train) | set(Y_test))
Xt = X_train.values
Y_t = Y_train.values
Xte = X_test.values
Y_te = Y_test.values
Yt = y2indicator(Y_t,K)
Ytet = y2indicator(Y_te,K)
from IrisChalenge import ANN
ann = ANN([6])
ann.fit(Xt,Yt,Xte,Ytet,epochs=1000,learning_rate=0.001,reg=0.01)
yhat = ann.predict(Xte)
print("Score :- ",np.mean(yhat==Y_te))
print("Score :- ",accuracy_score(Y_te,yhat))
print("Score :- ",confusion_matrix(Y_te,yhat))

