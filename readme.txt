
w2v
	- data-sets
	
	- dico_cleaning.py: deletes words present more than once
	
	- data_sort.py: sorts data-set to write sent_analysis input files
	
	- sent_analysis.py: performs binary classification with a word2vec based neural network

amazon_cells_labelled.txt: amazon labelled dataset

amazon_cells_labelled_w2v.txt: unlabelled amazon dataset 

amazon_cells_labelled_w2v.txt.zip: compressed unlabelled amazon data-set (input of word2vec_visualisation.py)

dicot.txt: optimized dictionary

dictionary_building.py: builds the optimized dictionary

forward_backward.py: 2 layer neural network (working with a random dictionary)

forward_backward_dic.py: 2 layer neural network (working with dico.txt)

imdb_labelled.txt: imdb data-set

logistic_regression.py: simple neural network (working with a random dictionary)

logistic_regression_dic.py: simple neural network (working with dico.txt)

word2vec_visualisation.py: computes the word2vec embedding graph

wordListAmazon.txt: contains all the amazon data-set words

yelp_labelled.txt: yelp data-set
