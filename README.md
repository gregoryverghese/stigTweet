Detecting Stigma towards Schizophrenia from Twitter using Neural Embedding Models
========================================================================================

This project explored diﬀerent language models to detect stigma towards Schizophrenia from Twitter posts. I build upon the study by Jilka et al using two other types of language representations and try to improve on a baseline bag of words model that scored 85% accuracy. Using neural word embeddings we achieve an accuracy of 89% with a Support Vector Machine (SVM) classiﬁer with the Word2Vec model. Finally, I explored deep contextualised word embeddings (BERT model) and get a maximum accuracy of 92% and an average accuracy of 90%. Furthermore, I ﬁnd that despite a signiﬁcantly smaller data set (13k tweets) the trained FastText and Word2Vec models perform better than the standard pre-trained GloVe model (trained on over 2 billion general subject tweets) which scores only 81%. I then explored character embedding models and found the best accuracy was only 71%. Finally, using vector arithmetic to represent word analogies I ﬁnd interesting results for my embedding vectors with intuitively satisfying answers to word analogies like the following

<p align="center">
    <strong> bipolar + medication = wit</strong>
</p>

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
10. SpaceyMoji: emoji tokenizer

following scripts contain classes/functions used across notebooks:

1.evaluation.py
2.evaluation1.py
3.feature_engineering.py
4.feature_engineering1.py
5.utility.py
6.ml.py
7.ml_config.py
8.preprocessing.py

Questions and Contact
--------------------

For any further questions feel free to drop me an email gregory.verghese@gmail.com

References
--------------------

[1] J. Sagar, C. Odoi, D. Wykes, and M. Cella, “Machine learning to detect mental health stigma on social media,” in prep.
