# -*- coding: utf-8 -*-
"""
Created on Sat Dec 15 16:07:39 2018

@author: Sabya
"""
from __future__ import print_function, division
from builtins import range
import numpy as np
import matplotlib.pyplot as plt
class Hidden(object):
    def __init__(self,fanin,fanout):
        self.W = np.random.randn(fanin,fanout)/np.sqrt(fanin+fanout)
        self.B = np.zeros(fanout)
    def relu(self,a):
        return a*(a>0)
    def forward(self,x):
        lin = x.dot(self.W)+self.B
        yhat = self.relu(lin)
        self.out = yhat
        return yhat
class ANN(object):
    def __init__(self,hiddenlayers):
        self.hidden = hiddenlayers
        self.hiddenlayerObj = []
    def cost(self,yhat,y):
        return -(y*np.log(yhat)).sum()
    def softmax(self,x):
        ex = np.exp(x)/np.exp(x).sum(axis=1,keepdims=True)
        return ex
    def hiddeninit(self,x_in):
        fanin = x_in.shape[1]
        for fanout in self.hidden:
            h = Hidden(fanin,fanout)
            self.hiddenlayerObj.append(h)
            fanin = fanout
    def forward(self,x_in):
        out = x_in
        for hid in self.hiddenlayerObj:
            out = hid.forward(out)
        outlin = out.dot(self.W)+self.B
        yhat = self.softmax(outlin)
        return yhat
    def grad(self,x_in,yhat,y,learning_rate,reg):
        delta = yhat - y
        gwo = self.hiddenlayerObj[-1].out.T.dot(delta)
        gdwo = gwo + (reg*self.W)
        gbo = delta.sum(axis=0)
        gdbo = gbo + (reg*self.B)
        self.W = self.W - (learning_rate*gdwo)
        self.B = self.B - (learning_rate*gdbo)
        for i in range(len(self.hiddenlayerObj)):
            i = i+1
            if i==1:
                delta = delta.dot(self.W.T)*(self.hiddenlayerObj[0-i].out>0)
            else:
                delta = delta.dot(self.hiddenlayerObj[0-(i-1)].W.T)*(self.hiddenlayerObj[0-i].out>0)
            if i == len(self.hiddenlayerObj):
                gwi = x_in.T.dot(delta)
            else:
                gwi = self.hiddenlayerObj[0-(i+1)].out.T.dot(delta)
            gbi = delta.sum(axis=0)
            gdwi = gwi + (reg*self.hiddenlayerObj[0-i].W)
            self.hiddenlayerObj[0-i].W = self.hiddenlayerObj[0-i].W - (learning_rate*gdwi)
            gdbi = gbi + (reg*self.hiddenlayerObj[0-i].B)
            self.hiddenlayerObj[0-i].B = self.hiddenlayerObj[0-i].B - (learning_rate*gdbi)
    def fit(self,x_train,y_train,x_test,y_test,learning_rate,reg,epochs=100):
        self.hiddeninit(x_train)
        self.W = np.random.randn(self.hidden[-1],y_train.shape[1])/np.sqrt(self.hidden[-1]+y_train.shape[1])
        self.B = np.zeros(y_train.shape[1])
        traincost = []
        testcost = []
        for epoch in range(epochs):
            yhat = self.forward(x_train)
            self.grad(x_train,yhat,y_train,learning_rate,reg)
            yhtest = self.forward(x_test)
            trainc = self.cost(yhat,y_train)
            testc = self.cost(yhtest,y_test)
            traincost.append(trainc)
            testcost.append(testc)
            print("Epoch No :- ",epoch," Train Cost :- ",trainc," Test Cost :- ",testc)
        plt.plot(traincost,label='Train cost')    
        plt.plot(testcost,label='Test cost')
        plt.legend()
        plt.show()
    def predict(self,x_in):
        yhat = self.forward(x_in)
        return np.argmax(yhat,axis=1)
        
###########################################
     #Testing Code   
##########################################  
#from process import get_data
#def y2indicator(y, K):
#    N = len(y)
#    ind = np.zeros((N, K))
#    for i in range(N):
#        ind[i, y[i]] = 1
#    return ind
#Xtrain, Y, Xtest, Yt = get_data()
#K = len(set(Y) | set(Yt))
#Ytrain = y2indicator(Y, K)
#Ytest = y2indicator(Yt, K)
#ann = ANN([20,30,15])
#ann.fit(Xtrain,Ytrain,Xtest,Ytest,learning_rate=0.0001,reg=0.01,epochs=6000)
#yhat = ann.predict(Xtest)
#print("Score :- ",np.mean(yhat==Yt))
#yhattr = ann.predict(Xtrain)
#print("TrainScore :-",np.mean(yhattr==Y))  
        
    
        
