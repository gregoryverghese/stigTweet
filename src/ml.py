from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import cross_val_predict
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import KFold, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
import pandas as pd
import numpy as np
from evaluation import Evaluation
from sklearn.model_selection import cross_val_predict, cross_validate
from sklearn.model_selection import RandomizedSearchCV


class baseML():
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels
        self.classifiers = None


    def getHyperParams(self, randomGrids, iterations=50, cv=5):

        clfParams = zip(self.classifiers, randomGrids)
        a = [RandomizedSearchCV(estimator = clf[0], param_distributions = clf[1], n_iter = iterations, cv = cv, verbose=2, random_state=42, n_jobs = -1) for clf in clfParams]
        params = [r.fit(self.features, self.labels) for r in a]
        return params

    def getTuningAccuracy(self, randomSearches):
        predictions = map(lambda f: f.predict(self.features), randomSearches)
        evalObj = Evaluation([self.labels]*len(predictions), predictions)
        accuracies = evalObj.getAccuracy()
        return accuracies

    def getCrossValidation(self, clf, parameters, k_fold=5):
        #return cross_val_predict(clf, self.features, self.labels, cv=k_fold, n_jobs = -1, fit_params=parameters)
        #return cross_val_score(clf, self.features, self.labels, cv=k_fold, n_jobs = -1)
        return cross_validate(clf, self.features, self.labels, scoring=['accuracy', 'f1', 'recall', 'precision', 'roc_auc'], cv=k_fold, n_jobs = -1)


    def getAllPredictions(self, methods=['RandomForest', 'GradientBoost', 'KNN', 'SVMLin', 'NaiveBayes'], parameters=None):
        predictions = map(self.getCrossValidation, self.classifiers, parameters)
        #evaluation =  self.getEval(predictions, methods)
        return predictions


    def trainAllClassifiers(self, methods=['RandomForest', 'GradientBoost', 'KNN', 'SVM', 'NaiveBayes']):
        self.classifiers = [getattr(self,'get'+f)() for f in methods]
        classifiersTrained = [clf.fit(self.features, self.labels) for clf in self.classifiers]
        return classifiersTrained

    def predictAllClassifiers(self, classifiers, test):
        predictions = [clf.predict(test) for clf in classifiers]
        return predictions

    def getEval(self, predict, methods=['RandomForest', 'GradientBoost', 'KNN', 'SVM', 'NaiveBayes']):
        evalObj = Evaluation([self.labels]*len(predict), predict)
        #results = eval('evalObj.get' + method + '()')
        results = evalObj.getSummary(methods)
        return results

    def getRandomForest(self, p=None):
        randFor = RandomForestClassifier(**p)
        return randFor

    def getGradientBoost(self, p=None):
        gradBoost = GradientBoostingClassifier(**p)
        return gradBoost

    def getKNN(self, p=None):
        knn = KNeighborsClassifier(**p)
        return knn

    def getSVM(self, p=None):
        svm = SVC(**p)
        return svm

    def getSVMPoly(self, p=None):
        svm = SVC(**p)
        return svm

    def getSVMSig(self, p=None):
        svm = SVC(**p)
        return svm

    def getNaiveBayes(self, p=None):
        nb = GaussianNB(**p)
        return nb
