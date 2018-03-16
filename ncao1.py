# -*- coding: utf-8 -*-
"""
Created on Mon Sep 26 18:32:40 2017
numerical computation and optimization-homework1
@author: 10152510119 徐紫琦
"""
import numpy as np

#given the shape of the matrix -> generate a random matrix with that shape
def generateRandom(m,n):
    return np.mat(np.random.random(size=(m,n)))

#1) perform the LU decomposition with the given A
#2) solve the Ax=b and return the solution x
def LU(A,b):
    #initialize the L matrix and the U matrix with the same shape as the A matrix
    #the L matrix is initialized to be a identity matrix with the intention of ensuring that all its diagonal elements equal to 1
    #the U matrix is initialized in a normal way: a square matrix whose elements all equal to 0
    n=len(A)
    L = np.mat(np.identity(n))
    U = np.mat(np.zeros((n,n)))
    #perform the decomposition
    #1) calculate the first column of the U matrix
    for i in range(0,n):
        U[(0,i)]=A[(0,i)]/L[(0,0)]
    #2) calculate the (i+1)-th comlumn of the L matrix and the (i+2)-th row in the U matrix
    for i in range(0,n-1):
        #calculate one by one from the (i+2)-th element to the last element in the (i+1)-th comlumn of the L matrix 
        for j in range(i+1,n):
            sum=0
            for k in range(0,n):
                if k!=i:
                    sum+=L[(j,k)]*U[(k,i)]
            L[(j,i)]=(A[(j,i)]-sum)/U[(i,i)]
        #calculate one by one from the (i+2)-th element to the last element in the (i+2)-th row in the U matrix
        for j in range(i+1,n):
            sum=0
            for k in range(0,n):
                if k!=i+1:
                    sum+=L[(i+1,k)]*U[(k,j)]
            U[(i+1,j)]=(A[(i+1,j)]-sum)
    #3) calculate the Y matrix which satisfies LY=b
    Y=np.mat(np.zeros((n,1)))
    for i in range(0,n):
        Y[i]=b[i]
        for j in range(0,i):
            Y[i]-=(L[(i,j)]*Y[j])
        Y[i]/=L[(i,i)]
    #4) solve the equation by calculating the X matrix which satisfies UX=Y
    X=np.mat(np.zeros((n,1)))
    for i in range(n-1,-1,-1):
        X[i]=Y[i]
        sum=0
        for j in range(i+1,n):
            X[i]-=U[(i,j)]*X[j]
        X[i]/=U[(i,i)]
    return X


def main():
    M=generateRandom(100,100)
    I=np.mat(np.identity(100,dtype="float"))
    A=M+I
    X=np.mat(range(1,101),dtype="float").transpose()
    b=np.dot(A,X)
    X1=LU(A,b)
    print("The X got from solving Ax=b:")
    print(X1)

if __name__=='__main__':
    main()
