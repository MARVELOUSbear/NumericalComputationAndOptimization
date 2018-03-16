# -*- coding: utf-8 -*-
"""
Created on Wed Dec 13 20:16:09 2017

@author: ALW15
"""
import numpy as np


def gradient_discent(A,b):
    X=np.mat(np.zeros((1000,1)))
    X0=X
    k=0
    eps=0.0001
    while True:
        r=np.mat((A.transpose()*A)*X-A.transpose()*b)#梯度方向
        q=np.mat(r.transpose()*r/(r.transpose()*A.transpose()*A*r))#每次迭代的步长
        X0=X
        X+=r*q#存储上一次X的值并更新X
        k+=1
        if np.linalg.norm(X-X0)<eps:#足够接近最优解后终止循环
            break
    return X,k

def accelerated_gradient_descent(A, b):
    X=np.mat(np.zeros((1000, 1)))#存储第K次迭代的X
    X0=X#存储第K-1次迭代的X
    X1=X#存储第K+1次迭代的X
    a0=0
    a1=1
    y=np.mat(X*(1+((a0-1)/a1))-X0*((a0-1)/a1))#根据公式计算出初始yk
    r=np.mat(A.transpose()*A*y-A.transpose()*b)#梯度方向，第一次迭代与之后相同
    L=np.mat(r.transpose()*r/(r.transpose()*A.transpose()*A*r))#迭代步长同样是第一次与之后相同
    X1=y-r*L
    k=1
    eps=0.0001
    while True:
        a0=a1#更新计算yk时X和X0的系数
        a1=(1+np.sqrt(1+4*(a0*a0)))/2
        X0=X
        X=X1
        y=X*(1+((a0-1)/a1))-X0*((a0-1)/a1)#更新yk
        r=A.transpose()*A*y-A.transpose()*b
        L=np.mat(r.transpose()*r/(r.transpose()*A.transpose()*A*r))
        X1=y-r*L#更新第K+1次迭代的X
        k+=1
        if np.linalg.norm(X1-X)<eps:
            break
    return X1,k

#代码运行结果有一点奇怪，加速梯度下降比梯度下降收敛慢很多很多，但是也没检查出来有什么问题
A=np.mat(np.random.random((1000,1000)))
X_op=np.mat(np.ones((1000,1)))
b=np.mat(np.dot(A,X_op))
X,k=gradient_discent(A,b)
print("The result X got from gradient discent:\n",X)
print(k," times of iteration in total")
X,k=accelerated_gradient_descent(A,b)
print("The result X got from accelerated gradient discent:\n",X)
print(k," times of iteration in total")


