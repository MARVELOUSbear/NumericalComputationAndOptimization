# -*- coding: utf-8 -*-
"""
numerical computation and optimization-homework3
@author: 10152510119 徐紫琦
"""

import numpy as np

def conjugate_gradient(A,b):
    n=A.shape[1]
    #initialize the value of x as a ramdom vector
    x0=np.mat(np.random.random((n,1)))
    x=x0
    r=np.mat(b-A*x)
    d=r
    for k in range(0,n-1):
        alpha=(r.transpose()*r)/(d.transpose()*A*d)
        x=x+alpha[(0,0)]*d
        r2=np.mat(b-A*x)
        #rules for termination
        if ((np.linalg.norm(r2)/np.linalg.norm(b)<=0.000001) or (k==n-1)):
            break
        beta=np.linalg.norm(r2)**2/np.linalg.norm(r)**2
        d=r2+beta*d
        r=r2
    return x
    
def QR(A,b):
    #A very simple implementation based on  Gram-Schmidt algorithm
    m,n=A.shape
    q=np.mat(np.zeros((m,n)))
    r=np.mat(np.zeros((m,n)))
    for k in range(0,n):
        s=0
        for j in range(0,m):
            s=s+A[(j,k)]**2
        r[(k,k)]=np.sqrt(s)    #update the value of the diagonal elements in r
        for j in range(0,m):
            q[(j,k)]=A[(j,k)]/r[(k,k)]    #get the value of elements in Q
        for i in range(k,n):
            s=0
            for j in range(0,m):
                s+=A[(j,i)]*q[(j,k)]
            r[(k,i)]=s    #get the value of elements in R
            for j in range(0,m):
                A[(j,i)]-=r[(k,i)]*q[(j,k)]#update the value of elements in A for further computation
    X=np.linalg.inv(r)*q.transpose()*b #get the value of x based on the equation: QX=Rb
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
    print("The standard result X got from np.linalg.solve:\n",np.linalg.solve(A,b))
    print("The result X got from conjugate gradient:\n",conjugate_gradient(A,b))
    print("The result X got from QR method:\n",QR(A,b))
    
if __name__=='__main__':
    main()