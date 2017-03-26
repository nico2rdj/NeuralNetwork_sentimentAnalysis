import sys
import collections
import sklearn.naive_bayes
import sklearn.linear_model
import nltk
import random
random.seed(0)
import numpy as np
from gensim.models.doc2vec import LabeledSentence, Doc2Vec
from sklearn.naive_bayes import BernoulliNB, GaussianNB
from sklearn.linear_model import LogisticRegression
nltk.download("stopwords")          # Download the stop words from nltk



path_to_data = './' 



def main():
    train_pos, train_neg, test_pos, test_neg = load_data(path_to_data)
    
    
    
    train_pos_vec, train_neg_vec, test_pos_vec, test_neg_vec = feature_vecs_DOC(train_pos, train_neg, test_pos, test_neg)
    lr_model = build_models_DOC(train_pos_vec, train_neg_vec)
	
    print("Logistic Regression")
    print("-------------------")
    evaluate_model(lr_model, test_pos_vec, test_neg_vec, True)
	
	

def load_data(path_to_dir):
    """
    Loads the train and test set into four different lists.
    """
    train_pos = []
    train_neg = []
    test_pos = []
    test_neg = []
    with open(path_to_dir+"train-pos.txt", "r") as f:
        for i,line in enumerate(f):
            words = [w.lower() for w in line.strip().split() if len(w)>=3]
            train_pos.append(words)
    with open(path_to_dir+"train-neg.txt", "r") as f:
        for line in f:
            words = [w.lower() for w in line.strip().split() if len(w)>=3]
            train_neg.append(words)
    with open(path_to_dir+"test-pos.txt", "r") as f:
        for line in f:
            words = [w.lower() for w in line.strip().split() if len(w)>=3]
            test_pos.append(words)
    with open(path_to_dir+"test-neg.txt", "r") as f:
        for line in f:
            words = [w.lower() for w in line.strip().split() if len(w)>=3]
            test_neg.append(words)
    return train_pos, train_neg, test_pos, test_neg

def remove_stop_words(user_input, stop_words):
    # Remove stopwords.
    purgedList = []
    for item in user_input:
        list = []
        for w in item:
           if w not in stop_words:
            list.append(w)
        purgedList.append(list)
    return purgedList



def feature_vecs_DOC(train_pos, train_neg, test_pos, test_neg):
    """
    Returns the feature vectors for all text in the train and test datasets.
    """
    # Doc2Vec requires LabeledSentence objects as input.
    # Turn the datasets from lists of words to lists of LabeledSentence objects.
    
    labeled_train_pos = []
    for i in range(len(train_pos)):
        labeled_train_pos.append(LabeledSentence(train_pos[i], [u'TRAIN_POS_' + str(i)]))	
    labeled_train_neg = []
    for i in range(len(train_neg)):
        labeled_train_neg.append(LabeledSentence(train_neg[i], [u'TRAIN_NEG_' + str(i)]))
    labeled_test_pos = []
    for i in range(len(test_pos)):
        labeled_test_pos.append(LabeledSentence(test_pos[i], [u'TEST_POS_' + str(i)]))
    labeled_test_neg = []
    for i in range(len(test_neg)):
        labeled_test_neg.append(LabeledSentence(test_neg[i], [u'TEST_NEG_' + str(i)]))
    # Initialize model
    model = Doc2Vec(dm=1, min_count=1, window=10, size=100, sample=1e-4, negative=5, workers=4)
    sentences = labeled_train_pos + labeled_train_neg + labeled_test_pos + labeled_test_neg
    model.build_vocab(sentences)
    # Train the model
    # This may take a bit to run 
    for i in range(100):
        #print("Training iteration %d" % (i))
        random.shuffle(sentences)
        model.train(sentences)
    # Use the docvecs function to extract the feature vectors for the training and test data
    
    train_pos_vec=[]
    for i in range(len(train_pos)):
        train_pos_vec.append(model.docvecs[u'TRAIN_POS_' + str(i)])
    train_neg_vec=[]
    for i in range(len(train_neg)):
        train_neg_vec.append(model.docvecs[u'TRAIN_NEG_' + str(i)])
    test_pos_vec=[]
    for i in range(len(test_pos)):
        test_pos_vec.append(model.docvecs[u'TEST_POS_' + str(i)])
    test_neg_vec=[]
    for i in range(len(test_neg)):
        test_neg_vec.append(model.docvecs[u'TEST_NEG_' + str(i)])
    # Return the four feature vectors
    return train_pos_vec, train_neg_vec, test_pos_vec, test_neg_vec



def build_models_DOC(train_pos_vec, train_neg_vec):
    """
    return LosticRegression Model that is fit to the training data.
    """
    Y = ["pos"]*len(train_pos_vec) + ["neg"]*len(train_neg_vec)
    # For LogisticRegression, pass no parameters
    
    lr_model = LogisticRegression()
    lr_model.fit(np.array(train_pos_vec + train_neg_vec), np.array(Y))
    return lr_model



def evaluate_model(model, test_pos_vec, test_neg_vec, print_confusion=False):
    """
    Prints the confusion matrix and accuracy of the model.
    """
    # Use the predict function and calculate the true/false positives and true/false negative.
    
    tp = 0
    fn = 0
    p_predict = model.predict(test_pos_vec)
    for item in p_predict:
        if item == "pos":
            tp = tp + 1
        else:
            fn = fn + 1
    n_predict = model.predict(test_neg_vec)
    fp = 0
    tn = 0
    for item in n_predict:
        if item == "neg":
            tn = tn + 1
        else:
            fp = fp + 1            
    accuracy = float(tp + tn) / (tp + tn + fn + fp)
    if print_confusion:
        print("predicted:\tpos\tneg")
        print("actual:")
        print("pos\t\t%d\t%d" % (tp, fn))
        print("neg\t\t%d\t%d" % (fp, tn))
    print("accuracy: %f" % (accuracy))



if __name__ == "__main__":
    main()