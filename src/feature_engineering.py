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


def preProcess(text):
    text = "".join([word.lower() for word in text if word not in string.punctuation])
    tokens = word_tokenize(text)
    text = [ps.stem(t) for t in tokens if t not in stopwords]
    return text

def clean_text(text):
    text = "".join([word.lower() for word in text if word not in string.punctuation])
    tokens = re.split('\W+', text)
    text = [ps.stem(word) for word in tokens if word not in stopwords]
    return text


def getTFIDVect(df1, df2, column, analyzer=clean_text, encoding=None, lowercase=None, stop_words=None):
    tfidfVect = TfidfVectorizer(analyzer, encoding, lowercase, stop_words)
    tfidf = tfidfVect.fit(df1[column])
    tfidfX = tfidf.transform(df2[column])
    return tfidfVect, tfidfX, tfidf


def getFeatureArray(df, cols1, tfidf, cols2):
    tfidfDf = pd.DataFrame(tfidf.toarray(), columns=cols2)
    featureVector = pd.concat([df[cols1].reset_index(drop=True), tfidfDf], axis=1)
    return featureVector


def gettfidfVectors(tweets, fCols):

    tfidfAll, tfidfX, tfidf = getTFIDVect(tweets, tweets, 'Tweet')
    xVectAll = getFeatureArray(tweets, fCols, tfidfX, tfidfAll.get_feature_names())

    return tfidfAll, xVectAll, tfidf


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

    '''
    def getFeatures(self):
        newDf = self.textDf
        newDf['Sentiment'] = newDf.apply(lambda x: tb(x).sentiment[0])
        newDf['Subjectivity'] = newDf.apply(lambda x: tb(x).subjectivity)
        newDf['Punc%'] = newDf.apply(lambda x: self.countPunc(x))
        newDf['AvgWord'] = newDf.apply(lambda x: self.getAvg(x))
        newDf['WordCount'] = newDf.apply(lambda x: len(str(x.encode('utf-8')).split(' ')))
        newDf['CharCount'] = newDf.apply(lambda x: len(x))
        newDf['hashtags'] = newDf.apply(lambda x: len([h for h in x if h=='#']))
        newDf['numerics'] = newDf.apply(lambda x: len([d for d in word_tokenize(x) if d.isdigit()]))
        newDf['upper'] = newDf.apply(lambda x: len([u for u in word_tokenize(x) if u.isupper()]))
        return newDf
    '''

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
