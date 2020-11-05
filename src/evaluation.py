from sklearn.model_selection import KFold, cross_val_score
import pandas as pd
import numpy as np
from gensim.models.word2vec import Word2Vec
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support as score


class Evaluation():

    def __init__(self, predict, labels):
        self.predict = predict
        self.labels = labels

    def getConfusion(self):
        return [confusion_matrix(l, p).ravel() for l, p in  zip(self.labels, self.predict)]

    def getTP(self):
        return map(lambda x: x[3], self.getConfusion())

    def getFP(self):
        return map(lambda x: x[1], self.getConfusion())

    def getTN(self):
        return map(lambda x: x[0], self.getConfusion())

    def getFN(self):
        return map(lambda x: x[2], self.getConfusion())

    def getAccuracy(self):
        return map(lambda tp, tn, fp, fn: ((tp+tn)/float(tp+tn+fp+fn)), self.getTP(),
                                                       self.getTN(), self.getFP(), self.getFN())

    def getMisclassification(self):
        return map(lambda tp, tn, fp, fn: ((fp+fn)/float(tp+tn+fp+fn)), self.getTP(),
                                                       self.getTN(), self.getFP(), self.getFN())

    def getPrecision(self):
        return map(lambda tp, fp: (tp/float(tp+fp)), self.getTP(), self.getFP())

    def getRecall(self):
        return map(lambda tp, fn: (tp/float(tp+fn)), self.getTP(), self.getFN())

    def getFScore(self, b=2):
        return map(lambda p, r: (1+b**2)*(p*r/((b**2*p) + r)), self.getPrecision(), self.getRecall())

    def getSpecificity(self):
        return map(lambda tn, fp: (tn/float(tn+fp)), self.getTN(), self.getFP())

    def getSensitivity(self):
        return map(lambda tp, fn: (tn/float(tp)), self.getTP(), self.getFN())

    def getROCurve(self):
        pass

    def getSummary(self, index):
        metrics = {'Accuracy': self.getAccuracy(),'Misclassification': self.getMisclassification(),'Precision': self.getPrecision(),
                                                               'Recall': self.getRecall(), 'F-Score': self.getFScore()}

        return pd.DataFrame(metrics, index=index)
