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


words = []
file = open('dico.txt','r')
while 1:
    line = file.readline()
    if not line:break
    string = line
    if line != '\n' and line != ' ' and line != '/':
        words.append(string[0:len(string)-1])
file.close()

for i in range(0,len(words)-1):
    for j in range(0,len(words)-1):
        if words[i] == words[j] and i != j:
            words[j] = ' '

file = open('dico.txt','w')
for i in range(0,len(words)-1):
    if words[i] != ' ':
        file.write(words[i])
        file.write('\n')
    
file.close()

"""
words = []
file = open('dico2.txt','r')
while 1:
    line = file.readline()
    if not line:break
    string = line
    if line != '\n' and line != ' ' and line != '/':
        words.append(string[0:len(string)-1])
file.close()

for i in range(0,len(words)-1):
    for j in range(0,len(words)-1):
        if words[i] == words[j] and i != j:
            words[j] = ' '

file = open('dico2.txt','w')
for i in range(0,len(words)-1):
    if words[i] != ' ':
        file.write(words[i])
        file.write('\n')
    
file.close()

words = []
file = open('dico3.txt','r')
while 1:
    line = file.readline()
    if not line:break
    string = line
    if line != '\n' and line != ' ' and line != '/':
        words.append(string[0:len(string)-1])
file.close()

for i in range(0,len(words)-1):
    for j in range(0,len(words)-1):
        if words[i] == words[j] and i != j:
            words[j] = ' '

file = open('dico3.txt','w')
for i in range(0,len(words)-1):
    if words[i] != ' ':
        file.write(words[i])
        file.write('\n')
    
file.close()


words = []
file = open('dico4.txt','r')
while 1:
    line = file.readline()
    if not line:break
    string = line
    if line != '\n' and line != ' ' and line != '/':
        words.append(string[0:len(string)-1])
file.close()

for i in range(0,len(words)-1):
    for j in range(0,len(words)-1):
        if words[i] == words[j] and i != j:
            words[j] = ' '

file = open('dico4.txt','w')
for i in range(0,len(words)-1):
    if words[i] != ' ':
        file.write(words[i])
        file.write('\n')
    
file.close()


words = []
file = open('dico5.txt','r')
while 1:
    line = file.readline()
    if not line:break
    string = line
    if line != '\n' and line != ' ' and line != '/':
        words.append(string[0:len(string)-1])
file.close()

for i in range(0,len(words)-1):
    for j in range(0,len(words)-1):
        if words[i] == words[j] and i != j:
            words[j] = ' '

file = open('dico5.txt','w')
for i in range(0,len(words)-1):
    if words[i] != ' ':
        file.write(words[i])
        file.write('\n')
    
file.close()


words = []
file = open('dico6.txt','r')
while 1:
    line = file.readline()
    if not line:break
    string = line
    if line != '\n' and line != ' ' and line != '/':
        words.append(string[0:len(string)-1])
file.close()

for i in range(0,len(words)-1):
    for j in range(0,len(words)-1):
        if words[i] == words[j] and i != j:
            words[j] = ' '

file = open('dico6.txt','w')
for i in range(0,len(words)-1):
    if words[i] != ' ':
        file.write(words[i])
        file.write('\n')
    
file.close()

words = []
file = open('dico7.txt','r')
while 1:
    line = file.readline()
    if not line:break
    string = line
    if line != '\n' and line != ' ' and line != '/':
        words.append(string[0:len(string)-1])
file.close()

for i in range(0,len(words)-1):
    for j in range(0,len(words)-1):
        if words[i] == words[j] and i != j:
            words[j] = ' '

file = open('dico7.txt','w')
for i in range(0,len(words)-1):
    if words[i] != ' ':
        file.write(words[i])
        file.write('\n')
    
file.close()


words = []
file = open('dico8.txt','r')
while 1:
    line = file.readline()
    if not line:break
    string = line
    if line != '\n' and line != ' ' and line != '/':
        words.append(string[0:len(string)-1])
file.close()

for i in range(0,len(words)-1):
    for j in range(0,len(words)-1):
        if words[i] == words[j] and i != j:
            words[j] = ' '

file = open('dico8.txt','w')
for i in range(0,len(words)-1):
    if words[i] != ' ':
        file.write(words[i])
        file.write('\n')
    
file.close()


words = []
file = open('dico9.txt','r')
while 1:
    line = file.readline()
    if not line:break
    string = line
    if line != '\n' and line != ' ' and line != '/':
        words.append(string[0:len(string)-1])
file.close()

for i in range(0,len(words)-1):
    for j in range(0,len(words)-1):
        if words[i] == words[j] and i != j:
            words[j] = ' '

file = open('dico9.txt','w')
for i in range(0,len(words)-1):
    if words[i] != ' ':
        file.write(words[i])
        file.write('\n')
    
file.close()


words = []
file = open('dico10.txt','r')
while 1:
    line = file.readline()
    if not line:break
    string = line
    if line != '\n' and line != ' ' and line != '/':
        words.append(string[0:len(string)-1])
file.close()

for i in range(0,len(words)-1):
    for j in range(0,len(words)-1):
        if words[i] == words[j] and i != j:
            words[j] = ' '

file = open('dico10.txt','w')
for i in range(0,len(words)-1):
    if words[i] != ' ':
        file.write(words[i])
        file.write('\n')
    
file.close()
"""