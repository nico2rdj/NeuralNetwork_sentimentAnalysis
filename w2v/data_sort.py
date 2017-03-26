from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import csv
import tensorflow as tf
from math import exp
import matplotlib.pyplot as plt
from random import randint 


# Data extraction 
x = np.array([])
file = open('imdb_labelled.txt','r')
string = ""
sentences_pos = []
sentences_neg = []
while 1:
    line = file.readline()
    if not line:break
    string = line
    if(int(string[len(string)-2])==1):
        sentences_pos.append(string[0:len(string)-3])
    else:
        sentences_neg.append(string[0:len(string)-3])
    
file.close()

file = open('train-pos.txt','w')
for i in range(0,len(sentences_pos)-100):
    file.write(sentences_pos[i])
    file.write('\n')
file.close()

file = open('test-pos.txt','w')
for i in range(400,len(sentences_pos)):
    file.write(sentences_pos[i])
    file.write('\n')
file.close()

file = open('train-neg.txt','w')
for i in range(0,len(sentences_neg)-100):
    file.write(sentences_neg[i])
    file.write('\n')
file.close()

file = open('test-neg.txt','w')
for i in range(400,len(sentences_neg)):
    file.write(sentences_neg[i])
    file.write('\n')
file.close()