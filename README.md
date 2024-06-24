# Detecting Stigma towards Schizophrenia from Twitter using Neural Embedding Models

This project explores different language representation models to detect stigma towards schizophrenia from Twitter posts. Building upon the study by Jilka et al., this work employs two additional types of language representations to improve on a baseline bag-of-words model that scored 85% accuracy. Using neural word embeddings, an accuracy of 89% was achieved with a Support Vector Machine (SVM) classifier with the Word2Vec model. Finally, using deep contextualized word embeddings (BERT), the model achieved maximum and average accuracies of 92% and 90%, respectively. Despite a significantly smaller dataset (13k tweets), the trained FastText and Word2Vec models outperformed the standard pre-trained GloVe model (trained on over 2 billion general subject tweets), which scored 81%. Character embedding models scored 71%. Interestingly, word embeddings can be interrogated using vector arithmetic to represent word analogies, such as:

<p align="center">
    <strong>bipolar + medication = wit</strong>
</p>

![Summary](data/summary.png?raw=true "Summary")

![Table Summary](data/table_summary.png?raw=true "Table Summary")

## Contents

The project contains the following 7 notebooks:

1. **preprocessing**: Data preprocessing 
2. **baseline**: TF-IDF code, feature engineering, and baseline classifiers
3. **embeddings**: Generates FastText and Word2Vec embeddings
4. **machine_learning**: ML classifiers
5. **neural_networks**: Neural network models
6. **bert**: PyTorch implementation for fine-tuning BERT model (adapted from Chris McCormick's blog)
7. **bert2**: PyTorch implementation for fine-tuning BERT model (unadapted from Chris McCormick's blog)
8. **analysis**: Analysis of results
9. **SpaceyMoji**: Emoji tokenizer

The following scripts contain classes/functions used across notebooks:

1. **evaluation.py**
2. **evaluation1.py**
3. **feature_engineering.py**
4. **feature_engineering1.py**
5. **utility.py**
6. **ml.py**
7. **ml_config.py**
8. **preprocessing.py**

## Todo

- [ ] Add ROC graphs
- [ ] Look at Reddit data

## Questions and Contact

For any further questions, feel free to drop me an email: [gregory.verghese@gmail.com](mailto:gregory.verghese@gmail.com)

## References

[1] J. Sagar, C. Odoi, D. Wykes, and M. Cella, “Machine learning to detect mental health stigma on social media,” in prep.

