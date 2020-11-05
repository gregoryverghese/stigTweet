import pandas as pd
import numpy as np
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer
import nltk
from gensim.models.phrases import Phrases, Phraser
from emoji import UNICODE_EMOJI

nltk.download('stopwords')
stopwords = nltk.corpus.stopwords.words('english')
nltk.download('wordnet')
lemmatizer = WordNetLemmatizer()
NOOCC_TOKEN = 'NOOCC'
MODEL_NUM=2


def getFile(fileName, column=None):
    socialDf = pd.read_csv(fileName, encoding='utf-8')
    return socialDf if column==None else socialDf[column]


def cleanScizAnnTwitter(annTwitter):

    annTwitter = annTwitter.dropna()
    annTwitter = annTwitter[annTwitter['Classification']!='o']
    annTwitter = annTwitter.astype({"Classification": int})
    annTwitter = annTwitter[annTwitter['Classification']!=1]
    annTwitter = annTwitter.replace(2, 1)

    return annTwitter


class SocialPreProcessing():
    def __init__(self, text, character):
        self.text = text
        self.character = character

    def clean(self, methods=['Tokens', 'Lemma']):
        for f in methods:
            print(f)
            self.text = self.text.apply(lambda x: getattr(self, 'get'+f)(x))
        return self.text

    def getTokens(self, sentences):
        tokens = word_tokenize(sentences) if not self.character else list(sentences)
        return tokens

    def getLemma(self, tokens):
        tokens = map(lemmatizer.lemmatize, tokens)
        return tokens

    def getStopwords(self, tokens):
        tokens = [t for t in tokens if t not in stopwords]
        return tokens

    def getEmoticons(self, tokens):
        return [t for t in tokens if t not in UNICODE_EMOJI]

    def getLowercase(self, tokens):
        return [t.lower() for t in tokens]

    def getPhrases(self, tokens):
        phrases = Phrases(tokens, min_count=1, threshold=1)
        bigrams = Phraser(phrases)
        text = [sent for sent in bigrams[tokens]]
        return tokens
