# schizophrenia-twitter

Project contains following 7 notebooks

1. 01 preprocessing.ipynb carries out initial cleaning of data, remove inconsistencies and nan values etc
2. 02 baseline.ipynb contains the code for the tf-idf model and classification
3. 03 embeddings.ipynb contains all the code related to generating FastText and Word2Vec models
4. 04 machine learning.ipynb contains code related to classification of embedding models
5. 05 neural networks.ipynb defines neural network models used for classification with embedding models
6. 06 bert.ipynb contains code for finetuning bert model in sentence classification task
7. 07 bert2.ipynb contains unadapted code for finetuning bert model in sentence classification task
8. 08 analysis.ipynb
9. 09 SpaceyMoji

following scripts

evaluation.py
evaluation1.py
feature_engineering.py
feature_engineering1.py
utility.py
ml.py
ml_config.py
Preprocessing.py

The scripts just contain classes or function that are used across notebooks and therefore it made sense to abstract out. The general workflow follows the numbering system. The dates included in the data folder, of which there are subfolders. Generally only interested in the files in the data/dataOut/Schiz/ path and data/baseline path. It is clear in the notebooks where data is being pulled from. There is also a embeddings folder where the embeddings are saved down. The code descriptions can be found in the appendix of the paper. General workflow is as follows

01 cleans the data
02 is all baseline model logic
03 here we apply NLP transformations, train all embedding models (Word2Vec and FastText and save them down
04 All machine learning code for classifying other than 1DCNN and LSTM live here
05 follows much the same logic as 04 but is concerned with neural networks
06 Bert Pytorch implementation, code provided Chris Mckormicks blog, url below
07 Bert Pytorch implementation 2, this is an unadapted version (minor adaption, my data and parameters)
08 analysis but more like sporadic analysis
09 SpaceyMoji used to tokenize the files, unfortunately I realised late on that NLTK does not properly tokenise all the emojis. SpaceyMoji is in Python3 Since the embedding and preprocessing logic is in Python2 I had to create a separate notebook

All scripts and workbooks are python 2 except notebooks 05, 06, 07 and 09.

Link to Pytorch implementation:

https://mccormickml.com/2019/07/22/BERT-fine-tuning/

For any further questions feel free to drop me an email gregory.verghese@gmail.com