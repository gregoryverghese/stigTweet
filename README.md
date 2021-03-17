Detecting Stigma towards Schizophrenia from Twitter using Neural Embedding Models
========================================================================================

This project explored diﬀerent language representation models to detect stigma towards Schizophrenia from Twitter posts. I build upon the study by Jilka et al using two other types of language representations and try to improve on a baseline bag of words model that scored 85% accuracy. Using neural word embeddings we achieve an accuracy of 89% with a Support Vector Machine (SVM) classiﬁer with the Word2Vec model. Finally, using, deep contextualised word embeddings (BERT) the model achieved 92% and 90% for maxmium and average accuracy respectively. In addition, despite a signiﬁcantly smaller data set (13k tweets) the trained FastText and Word2Vec models performed better than the standard pre-trained GloVe model (trained on over 2 billion general subject tweets) which scored 81%. Character embedding models scored 71%. Interestingly, word embeddings can be interogated using vector arithmetic to represent word analogies such as.

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

1. evaluation.py
2. evaluation1.py
3. feature_engineering.py
4. feature_engineering1.py
5. utility.py
6. ml.py
7. ml_config.py
8. preprocessing.py

Todo
--------------------

1. add roc graphs
2. look at reddit data

Questions and Contact
--------------------

For any further questions feel free to drop me an email gregory.verghese@gmail.com

References
--------------------

[1] J. Sagar, C. Odoi, D. Wykes, and M. Cella, “Machine learning to detect mental health stigma on social media,” in prep.
