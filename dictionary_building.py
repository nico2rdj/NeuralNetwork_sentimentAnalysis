"""  COGS_181 
----FINAL PROJECT-----
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import csv
import tensorflow as tf
from math import exp
import matplotlib.pyplot as plt
from random import randint 
#-----------------
# Function definition
#-----------------

def f(w,b, x):
    if (1/(1+exp(-np.dot(w.transpose(),x)-b)) >= 0.5):
        return 1;
    else:
        return 0;
        
def error(X, Y, w, b):
    sum_ = 0
    for i in range(0,len(Y)-1):
        if (f(w, b, X[i,:])!=Y[i]):
            sum_+=1
    sum_ = sum_ / len(Y)
    return sum_;

def loss(x,y,w,b):
    sum_ = 0
    for i in range(0,len(x)-1):
        xi = x[i]
        yi = y[i]
        sum_+= yi*np.log(f(w,b,xi))+(1-yi)*np.log(1-f(w,b,xi))
    return -sum_;
            

def Gradw(X,Y,w,b):
    grad = 0
    for i in range(0,len(Y)-1):
        grad += -X[i,:] + X[i,:]*exp(-np.dot(w.transpose(),X[i,:])-b)/(1+exp(-np.dot(w.transpose(),X[i,:])-b)) + Y[i]*X[i,:]
    return -grad;

def Gradb(X,Y,w,b):
    grad = 0
    for i in range(0,len(Y)-1):
        grad += -1 + exp(-np.dot(w.transpose(),X[i,:])-b)/(1+exp(-np.dot(w.transpose(),X[i,:])-b)) + Y[i] 
    return -grad;


# Data extraction 
file_path = 'amazon_cells_labelled.txt'
x = np.array([])
file = open('amazon_cells_labelled.txt','r')
string = ""
labels = []
sentences = []
while 1:
    line = file.readline()
    if not line:break
    string = line
    sentences.append(string[0:len(string)-3])
    labels.append(int(string[len(string)-2]))
file.close()
 
# Data treatment
# Fill the list of words
words = []
file = open('wordListAmazon.txt','r')
while 1:
    line = file.readline()
    if not line:break
    string = line
    if line != '\n' and line != ' ' and line != '/':
        words.append(string[0:len(string)-1])
file.close()

# Select 1000 words
for kit in range(0,10):
    if(kit!=9):
        num = kit*1000
        wordShuffled = words[num:num+999] 
    else:
        num = kit*1000
        wordShuffled = words[num:len(words)-1]

    # One hot encoding of the sentences
    xFinal = []        
    for i in range(0,len(sentences)):
        xTemp = np.zeros(len(wordShuffled))
        for k in range(0,len(wordShuffled)-1):
            for j in range(0,len(sentences[i].split())-1):
                if (wordShuffled[k]==sentences[i].split()[j]):
                    xTemp[k] = 1
        xFinal.append(xTemp)

    # Train and test sets
    x_train = np.array(xFinal)[0:799]
    y_train = labels[0:799]
    x_test = np.array(xFinal)[800:999]
    y_test = labels[800:999]

    #-----------------
    # weight and bias initialization
    #-----------------
    print(len(wordShuffled))
    w = np.zeros(len(wordShuffled)) #((x_train.shape[1])) 
    b = 0

    #-----------------
    # While loop
    #-----------------
    
    j = 1
    Lambda = 1e-3
    abs = []
    ord_train = []
    ord_test = []
    n = 10000
    while (j<=n):
        # Select random xi, yi
        ord_train.append(1-error(x_train, y_train, w, b))
        ord_test.append(1-error(x_test, y_test, w, b))
        w = w - Lambda*Gradw(x_train,y_train,w,b)
        b = b - Lambda*Gradb(x_train,y_train,w,b) 
        j+=1    
                
                
    #plt.figure(1)
    #plt.axis([0,n,0,1])
    #plt.plot(abs,ord_train, label="Training accuracy") 
    #plt.plot(abs,ord_test, label = "Test accuracy")
    #plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

    #-----------------
    # Accuracy
    #-----------------
    err = error(x_train, y_train, w, b)
    print("accuracy training",1-err) 
    err = error(x_test, y_test, w, b)
    print("accuracy test",1-err) 
    
    if(kit==0):
        file = open('dico1.txt','w')
    if(kit==1):
        file = open('dico2.txt','w')
    if(kit==2):
        file = open('dico3.txt','w') 
    if(kit==3):
        file = open('dico4.txt','w')
    if(kit==4):
        file = open('dico5.txt','w')
    if(kit==5):
        file = open('dico6.txt','w')
    if(kit==6):
        file = open('dico7.txt','w')
    if(kit==7):
        file = open('dico8.txt','w')
    if(kit==8):
        file = open('dico9.txt','w')
    if(kit==9):
        file = open('dico10.txt','w')
        
    for i in range(0,len(w)):
        if w[i]>0 or w[i]<-0:
            file.write(words[num+i])
            file.write('\n')
    file.close()
    
    print(kit, 'dico fait')


        
        
        
        
        


