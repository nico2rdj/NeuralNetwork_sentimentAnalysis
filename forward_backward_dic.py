import numpy as np
from tqdm import *
import matplotlib.pyplot as plt


def sigmoid(x):
    return 1.0/(1+np.exp(-x))

def crossentropy(f,y):
    return -np.sum(y*np.log(f)+(1-y)*np.log(1-f))

# Data extraction 
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
train_err = []
test_err = []
x_train = np.array(xFinal)[0:799]
y_train = labels[0:799]
y_train = np.array(y_train)
num_train = len(x_train)
y_train = y_train.reshape(num_train,1)
x_test = np.array(xFinal)[800:999]
y_test = labels[800:999]
y_test = np.array(y_test)
num_test = len(x_test)
y_test = y_test.reshape(num_test,1)

#-----------------
# weight and bias initialization
#-----------------
J=2
n_iter = 15000
alpha = 0.002
w1 = np.random.randn(len(words),J)/((len(words)*J)**2) 
w2 = np.random.randn(J+1,1)/((J+1)**2)
dw1_ = []
train_loss = []
for n in range(n_iter):
    # forward computation
    q = np.copy(sigmoid(np.dot(x_train,w1)))
    qe = np.concatenate((np.ones((num_train,1)),q),axis=1)
    y = np.copy(sigmoid(np.dot(qe,w2)))
    # backward computation
    temp = y_train-y
    dqe = np.dot(temp,w2.T)
    dw2 = np.dot(qe.T,temp)
    dq = dqe[:,1:J+1]
    dw1 = np.dot(x_train.T,dq*(1-q)*q) # weight updating
    w1 = w1 + alpha*dw1
    w2 = w2 + alpha*dw2
    # training error
    predict = y >= 0.5
    train_err.append(np.sum(predict == y_train)*1.0/num_train)
    # training loss
    train_loss.append(crossentropy(y,y_train))
    # test error
    q = np.copy(sigmoid(np.dot(x_test,w1)))
    q_ = np.concatenate((np.ones((num_test,1)),q),axis=1)
    y = sigmoid(np.dot(q_,w2))
    predict = y >= 0.5
    test_err.append(np.sum(predict == y_test)*1.0/num_test)

print('test accuracy',test_err[len(test_err)-1])
print('train accuracy',train_err[len(train_err)-1])

plt.figure(1)
plt.plot(train_err, label="Training accuracy") 
plt.plot(test_err, label = "Test accuracy")
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.figure(2)
plt.plot(train_loss, label ="Training loss")
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)