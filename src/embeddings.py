from gensim.scripts.glove2word2vec import glove2word2vec
from gensim.models.keyedvectors import KeyedVectors
from gensim.models import FastText
from gensim.models.word2vec import Word2Vec
import pandas as pd
import numpy as np
import gensim as g

NOOCC_INDEX = 0
NOOCC_TOKEN = 'NOOCC'



class EmbeddingModel():
    def __init__(self, library=None, tokens=None, name=None, model=None):
        self.lib = library
        self.tokens = tokens
        self.name = name
        self.model = model

    def getEmbeddings(self, trainMethod, emArgs):

        model = self.lib(min_count=emArgs[0], window=emArgs[1], size=emArgs[2], sample=emArgs[3], alpha=emArgs[4], min_alpha=emArgs[5], negative=emArgs[6], workers=20, sg=trainMethod)
        model.build_vocab(self.tokens, progress_per=10000)
        model.train(self.tokens, total_examples=model.corpus_count, epochs=30, report_delay=1)
        model.init_sims(replace=True)
        self.save(model)
        self.model = model

    def save(self, model):
        model.save(self.name)

    def load(self, path):
        self.model = self.lib.load(path + self.name)

    def getSimilar(self, model, word):
        return self.model.wv.most_similar(positive=[word])

    def getWordIndex(self, newWord=NOOCC_TOKEN, newIndex=NOOCC_INDEX ):

        wordIndex = {k: v.index for k, v in self.model.wv.vocab.items()}
        self.model.wv.vectors = np.insert(self.model.wv.vectors, [newIndex], self.model.wv.vectors.mean(0), axis=0)
        wordIndex = {word:(index+1) if index>= newIndex else index for word, index in wordIndex.items()}
        wordIndex[newWord] = newIndex

        return wordIndex

    def getIndexData(self, xText, labels, wordIndex):
        xTrain = [[wordIndex[tok] if tok in wordIndex else wordIndex[NOOCC_TOKEN] for tok in s] for s in xText]
        return (np.array(xTrain), np.array(labels))

    def countMissing(self, text, wordIndex):
        return sum([1 for s in text for tok in s if tok not in wordIndex])
