Detecting Stigma towards Schizophrenia from Twitter using Neural Embedding Models
========================================================================================

This project explored diﬀerent language models to detect stigma towards Schizophrenia from Twitter posts. I build upon the study by Jilka et al using two other types of language representations and try to improve on a baseline bag of words model that scored 85% accuracy. Using neural word embeddings we achieve an accuracy of 89% with a Support Vector Machine (SVM) classiﬁer with the Word2Vec model. Finally, I explored deep contextualised word embeddings (BERT model) and get a maximum accuracy of 92% and an average accuracy of 90%. Furthermore, I ﬁnd that despite a signiﬁcantly smaller data set (13k tweets) the trained FastText and Word2Vec models perform better than the standard pre-trained GloVe model (trained on over 2 billion general subject tweets) which scores only 81%. I then explored character embedding models and found the best accuracy was only 71%. Finally, using vector arithmetic to represent word analogies I ﬁnd interesting results for my embedding vectors with intuitively satisfying answers to word analogies like the following

<p align="center">
    **bipolar + medication = wit**
</p>
> 
> 
![Alt text](data/summary.png?raw=true "Title")

![Alt text](data/table_summary.png?raw=true "Title")

Contents
--------

The project contains following 7 notebooks:

1. preprocessing: data preprocessing 
2. baseline: tf-idf code, feature engineering and baseline classifiers
3. embeddings: generates FastText and Word2Vec embeddings
4. machine learning: ml classifiers
5. neural networks: neural network models
6. bert: Pytorch implementation for finetuning Bert model (adapted from Chris Mckormicks blog)
7. bert2: Pytorch implementation for finetuning Bert model (unadapted from Chris Mckormicks blog)
9. analysis:
10. SpaceyMoji: SpaceyMoji to tokenize emojis

following scripts contain classes/functions used across notebooks:

evaluation.py
evaluation1.py
feature_engineering.py
feature_engineering1.py
utility.py
ml.py
ml_config.py
Preprocessing.py

Questions and Contact
--------------------

For any further questions feel free to drop me an email gregory.verghese@gmail.com

References
--------------------

[1] J. Sagar, C. Odoi, D. Wykes, and M. Cella, “Machine learning to detect mental health stigma on social media,” in prep.
