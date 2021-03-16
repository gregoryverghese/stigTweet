Detecting Stigma towards Schizophrenia from Twitter using Neural Embedding Models
========================================================================================

This project explored diﬀerent language models to detect stigma towards Schizophrenia from Twitter posts. I build upon the study by Jilka et al using two other types of language representations and try to improve on a baseline bag of words model that scored 85% accuracy. Using neural word embeddings we achieve an accuracy of 89% with a Support Vector Machine (SVM) classiﬁer with the Word2Vec model. Finally, I explored deep contextualised word embeddings (BERT model) and get a maximum accuracy of 92% and an average accuracy of 90%. Furthermore, I ﬁnd that despite a signiﬁcantly smaller data set (13k tweets) the trained FastText and Word2Vec models perform better than the standard pre-trained GloVe model (trained on over 2 billion general subject tweets) which scores only 81%. I then explored character embedding models and found the best accuracy was only 71%. Finally, using vector arithmetic to represent word analogies I ﬁnd interesting results for my embedding vectors with intuitively satisfying answers to word analogies like the following

> bipolar + medication = wit
> 
![Alt text](data/summary.png?raw=true "Title")

Contents
--------

The project contains following 7 notebooks:

1. preprocessing.ipynb carries out initial cleaning of data, remove inconsistencies and nan values etc
2. baseline.ipynb contains the code for the tf-idf model and classification
3. embeddings.ipynb contains all the code related to generating FastText and Word2Vec models
4. machine learning.ipynb contains code related to classification of embedding models
5. neural networks.ipynb defines neural network models used for classification with embedding models
6. bert.ipynb contains code for finetuning bert model in sentence classification task
7. bert2.ipynb contains unadapted code for finetuning bert model in sentence classification task
8. analysis.ipynb
9. SpaceyMoji

and the following scripts:

evaluation.py
evaluation1.py
feature_engineering.py
feature_engineering1.py
utility.py
ml.py
ml_config.py
Preprocessing.py

The scripts just contain classes or function that are used across notebooks and therefore it made sense to abstract out. The general workflow follows the numbering system. The dates included in the data folder, of which there are subfolders. Generally only interested in the files in the data/dataOut/Schiz/ path and data/baseline path. It is clear in the notebooks where data is being pulled from. There is also a embeddings folder where the embeddings are saved down. The code descriptions can be found in the appendix of the paper. General workflow is as follows

1. data preprocessing
2. baseline model - tf-idf code and feature engineering
3. NLP transformations, train all embedding models (Word2Vec and FastText and save them down
04 All machine learning code for classifying other than 1DCNN and LSTM live here
05 follows much the same logic as 04 but is concerned with neural networks
06 Bert Pytorch implementation, code provided Chris Mckormicks blog, url below
07 Bert Pytorch implementation 2, this is an unadapted version (minor adaption, my data and parameters)
08 analysis but more like sporadic analysis
09 SpaceyMoji used to tokenize the files, unfortunately I realised late on that NLTK does not properly tokenise all the emojis. SpaceyMoji is in Python3 Since the embedding and preprocessing logic is in Python2 I had to create a separate notebook

All scripts and workbooks are python 2 except notebooks 05, 06, 07 and 09.


Questions and Contact
--------------------

For any further questions feel free to drop me an email gregory.verghese@gmail.com

[1]
