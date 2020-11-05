import pandas as pd
import re
import nltk
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
import string
import numpy as np
from textblob import TextBlob as tb

from sklearn.model_selection import train_test_split

ps = nltk.PorterStemmer()
nltk.download('punkt')
nltk.download('stopwords')
stopwords = nltk.corpus.stopwords.words('english')


class FeatureEngineering():

    def __init__(self, textDf):
        self.textDf = textDf

    def countPunc(self, sentence):
        punc = list(filter(lambda x: x[0] in string.punctuation, sentence))
        punc = len(punc)/float((len(sentence.replace(' ', ''))))
        return (punc * 100)

    def getAvg(self, sentence):
        words = sentence.split()
        words = list(filter(lambda x: x not in string.punctuation, words))
        length = [len(w) for w in words]
        return sum(length)/float(len(words))

    def getFeatures(self, column):
        newDf = self.textDf
        newDf['Sentiment'] = newDf[column].apply(lambda x: tb(x).sentiment[0])
        newDf['Subjectivity'] = newDf[column].apply(lambda x: tb(x).subjectivity)
        newDf['Punc%'] = newDf[column].apply(lambda x: self.countPunc(x))
        newDf['AvgWord'] = newDf[column].apply(lambda x: self.getAvg(x))
        newDf['WordCount'] = newDf[column].apply(lambda x: len(str(x.encode('utf-8')).split(' ')))
        newDf['CharCount'] = newDf[column].apply(lambda x: len(x))
        newDf['hashtags'] = newDf[column].apply(lambda x: len([h for h in x if h=='#']))
        newDf['numerics'] = newDf[column].apply(lambda x: len([d for d in word_tokenize(x) if d.isdigit()]))
        newDf['upper'] = newDf[column].apply(lambda x: len([u for u in word_tokenize(x) if u.isupper()]))
        return newDf
