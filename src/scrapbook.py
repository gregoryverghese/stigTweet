#Get tfidf vectors for the whole data set to be used in cross validation. Also get tfidf vectors for test and train sets
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


def getTFIDVect(df1, df2, column):
    tfidfVect = TfidfVectorizer(analyzer=clean_text)
    tfidf = tfidfVect.fit(df1[column])
    tfidf = tfidf.transform(df2[column])
    return tfidfVect, tfidf

def getFeatureArray(df, cols1, tfidf, cols2):
    tfidfDf = pd.DataFrame(tfidf.toarray(), columns=cols2)
    featureVector = pd.concat([df[cols1].reset_index(drop=True), tfidfDf], axis=1)
    return featureVector


def gettfidfVectors(xTrain, xTest, tweets, fCols):

    tfidfTrain, xTrainTFIDF= getTFIDVect(xTrain, xTrain, 'Tweet')
    xVectTrain = getFeatureArray(xTrain, fCols, xTrainTFIDF, tfidfTrain.get_feature_names())

    tfidfTest, xTestTFIDF = getTFIDVect(xTrain, xTest, 'Tweet')
    xVectTest = getFeatureArray(xTest, fCols, xTestTFIDF, tfidfTest.get_feature_names())

    tfidfAll, tfidfXAll = getTFIDVect(tweets, tweets, 'Tweet')
    xVectAll = getFeatureArray(tweets, fCols, tfidfXAll, tfidfAll.get_feature_names())

    return xVectAll
