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
x = np.array([])
file = open('imdb_labelled.txt','r') 
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

# Fill the list of words
words = []
file = open('dico.txt','r')
while 1:
    line = file.readline()
    if not line:break
    string = line
    if line != '\n' and line != ' ':
        words.append(string[0:len(string)-1])
file.close()
"""file = open('positiveWords.txt','r')
while 1:
    line = file.readline()
    if not line:break
    string = line
    if line != '\n' and line != ' ' and line != '/':
        words.append(string[0:len(string)-1])
file.close()

file = open('negativeWords.txt','r')
while 1:
    line = file.readline()
    if not line:break
    string = line
    if line != '\n' and line != ' ' and line != '/':
        words.append(string[0:len(string)-1])
file.close()"""

for i in range(0,len(words)):
    words[i] = words[i].lower()

# One hot encoding of the sentences
xFinal = []        
for i in range(0,len(sentences)):
    xTemp = np.zeros(len(words))
    for k in range(0,len(words)-1):
        for j in range(0,len(sentences[i].split())-1):
            if (words[k]==sentences[i].split()[j]):
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

w = np.zeros(len(words)) #((x_train.shape[1])) 
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
    """i = randint(0,len(x_train)-1) 
    if f(w, b, x_train[i,:])==y_train[i]:
        abs.append(j)
        ord_train.append(1-error(x_train, y_train, w, b))
        ord_test.append(1-error(x_test, y_test, w, b))
        j+=1
    else:"""
    abs.append(j)
    ord_train.append(1-error(x_train, y_train, w, b))
    ord_test.append(1-error(x_test, y_test, w, b))
    w = w - Lambda*Gradw(x_train,y_train,w,b)
    b = b - Lambda*Gradb(x_train,y_train,w,b) 
    j+=1    

plt.figure(1)
plt.axis([0,n,0,1])
plt.plot(abs,ord_train, label="Training accuracy") 
plt.plot(abs,ord_test, label = "Test accuracy")
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

#-----------------
# Accuracy
#-----------------
err = error(x_train, y_train, w, b)
print("accuracy training",1-err) 
err = error(x_test, y_test, w, b)
print("accuracy test",1-err) 

 


        
        
        
        
        


