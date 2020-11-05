import pandas as pd
import numpy as np
from nltk import word_tokenize

abbreviations = pd.read_csv('data/other/abbreviations.csv')['Abbreviation'].tolist()
abbreviations = [str(a).strip() for a in abbreviations]

def getFilePath(path1, path2, names):
    paths = getPaths(path1, path2, names)

    return [p + n for p, n in zip(paths, names)]


def getParameters(parameters, models):
    parameters = [[p for p in parameters] for m in models]
    return parameters


def getPaths(path1, path2, models):
    return path1*(len(models)/2) + path2*(len(models)/2)


def getTokenTypes(token, models):
    return [token]*len(models)


def getCleanSchedule(clean):
    pass


def getBestParameters(model):
    return [m.best_params_ for m in model[0][0]]


def getEvalDf(results, name, idx, methods):

    accuracy = [clf['test_accuracy'].mean() for clf in results[idx]]
    precision = [clf['test_precision'].mean() for clf in results[idx]]
    recall = [clf['test_recall'].mean() for clf in results[idx]]
    f1 = [clf['test_f1'].mean() for clf in results[idx]]
    roc_auc = [clf['test_roc_auc'].mean() for clf in results[idx]]

    evalDict = {'accuracy':accuracy, 'precision':precision, 'recall':recall, 'f1':f1, 'roc_auc':roc_auc}
    evalDf = pd.DataFrame(evalDict, index=methods)

    evalDf.to_csv(name + '.csv')

    return evalDf


def getAverage(results, metric):

    classifiers = zip(*results)
    average = [np.array([trial[metric].mean() for trial in clf]).mean() for clf in classifiers]

    return average


def getEvalTrials(results, names, path):

    results = zip(*results)

    for m in len(range(results)):
        accuracy = getAverage(results[m], 'test_accuracy')
        precision = getAverage(results[m], 'test_precision')
        recall = getAverage(results[m], 'test_recall')
        f1 = getAverage(results[m], 'test_f1')
        roc_auc = getAverage(results[m], 'test_roc_auc')

        evalDict = {'accuracy':accuracy, 'precision':precision, 'recall':recall, 'f1':f1, 'roc_auc':roc_auc}
        evalDf = pd.DataFrame(evalDict, index=methods)

        evalDf.to_csv(path + names[m] + '.csv')


def getEmoticonText(tokens, emojis):

    emojiLines = list(map(lambda x: any(i in emojis for i in x), tokens))
    tokens = np.array(tokens)
    emojiLines = np.array(emojiLines)

    tokens = tokens[emojiLines]

    return tokens


def getLowerText(tokens):

    lowerLines = list(filter(lambda x: any(i.isupper() and len(i) > 1
                                           and i != 'RT' and i not in abbreviations for i in x), tokens))
    tokens = np.array(tokens)
    lowerLines = np.array(lowerLines)

    return lowerLines


def getModelsTwo(embeddingText, emojis=None):

    tokens = embeddingText.apply(lambda x: word_tokenize(x))
    tokens = tokens.tolist()
    tokens = getLowerText(tokens) if emojis == None else getEmoticonText(tokens, emojis)
    text = pd.Series(tokens)
    text = text.apply(lambda x: ' '.join(x))

    return text
