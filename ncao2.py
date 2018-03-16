# -*- coding: utf-8 -*-
"""
numerical computation and optimization-homework2
@author: 10152510119 徐紫琦
"""
import numpy as np

#generate the three matrix used in different iterative methods
def generateLUD(A):
    D=np.mat(np.diag(np.diag(A)))
    L=np.mat((-1)*np.tril(A,k=-1))
    U=np.mat((-1)*np.triu(A,k=1))
    return D,L,U

#implementation of jacobi iterative method
def jacobi(A,b):
    X=np.zeros((100,1))
    X0=X
    D,L,U=generateLUD(A)
    #B=(D)^(-1)*(L+U)
    B=np.linalg.inv(np.mat(D))*np.mat(L+U)
    #f=(D)^(-1)*(b)
    f=np.linalg.inv(np.mat(D))*np.mat(b)
    #X=B*X+f <-keep iterating until X converges to the final result
    while True:
        X=np.dot(B,X0)+f
        if np.linalg.norm(np.dot(A,X)-b)/np.linalg.norm(b)<0.000001:
            break
        X0 = X
    return X

#implementation of Gauss-Seidel iterative method
def gaussian(A,b):
    X=np.zeros((100,1))
    X0=X
    D,L,U=generateLUD(A)
    #B=(D-L)^(-1)*(U)
    B=np.linalg.inv(np.mat(D-L))* np.mat(U)
    #f=(D-L)^(-1)*(b)
    f=np.linalg.inv(np.mat(D-L))*np.mat(b)
    #X=B*X+f <-keep iterating until X converges to the final result
    while True:
        X=np.dot(B,X0)+f
        if np.linalg.norm(np.dot(A,X)-b)/np.linalg.norm(b)<0.000001:
            break
        X0=X
    return X

#implementation of successive overrelaxation method
def SOR(A,b):
    D,L,U=generateLUD(A)
    w=1.05
    X=np.mat(np.zeros((100,1)))
    X0=X
    #B=(D-wL)^(-1)*[(1-w)D+wU
    B=np.dot(np.linalg.inv(D-w*L),(1-w)*D+w*U)
    #f=w(D-wL)^(-1)*b
    f=np.dot(w*np.linalg.inv(D-w*L),b)
    #X=B*X+f <-keep iterating until X converges to the final result
    while True: 
        X=np.dot(B,X0)+f
        if np.linalg.norm(A*X0-b)/np.linalg.norm(b)<0.000001:
            break
        X0=X
    return X  
    
        
def main():
    A=np.mat(np.zeros((100,100)))
    for i in range(0,100):
        if i>0:
            A[(i,i-1)]=-1
        if i<99:
            A[(i,i+1)]=-1
        A[(i,i)]=2
    b=np.mat(np.ones((100,1)))
    print("Standard result got from np.linalg.solve:")
    print(np.linalg.solve(A,b))
    print("Result of Jacobi Iterative Method:")
    print(jacobi(A,b))
    print("Result of Gauss-Seidel Iterative Method:")
    print(gaussian(A,b))
    print("Result of SOR Method:")
    print(SOR(A,b))

if __name__=='__main__':
    main()
