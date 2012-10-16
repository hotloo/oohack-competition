#!/usr/bin/python2
from __future__ import division
from sklearn.ensemble import RandomForestClassifier
# from sklearn import svm
# from sklearn.naive_bayes import MultinomialNB
# from sklearn import neighbors
# from sklearn import linear_model
# from sklearn import decomposition
from sklearn import preprocessing
from sklearn import cross_validation
import numpy as np


# Additional preprocessing ideas: if a company is dead treat it as false even if
# it's a positive example?


def normalize_nan(X, columns):
    '''Values that are None (missing) are represented by np.nan in in a
    hardcoded way. This puts mean of that column instead.'''
    masked_X = np.ma.masked_array(X, np.isnan(X))
    for i in columns:
        X[:, i] = masked_X[:, i].filled(np.mean(masked_X, axis=0)[i])


def normalize_hashes(X, columns):
    '''Some string values are represented as hash integers, this makes those
    values continuous.'''
    le = preprocessing.LabelEncoder()
    for i in columns:
        X[:, i] = le.fit_transform(X[:, i])


def evaluate(target, predicted):
    total = len(target)
    correct = sum(target == predicted)
    accuracy = correct / total
    return accuracy


def main():
    print 'Loading train data'
    dataset = np.load('../data/oohack_train.npy')
    # print 'Loading test data'
    # testset = np.load('../data/oohack_test.npy')

    data_target = np.array([x[0] for x in dataset])
    data = np.array([x[1:] for x in dataset])
    # test_target = np.array([x[0] for x in testset])
    # test = np.array([x[1:] for x in testset])

    print 'Data loaded, starting pre-processing'
    #Dirty, no time
    indexes = []
    others = []
    i = 0
    while i < len(data_target):
        if data_target[i] == 1:
            indexes.append(i)
        else:
            others.append(i)
        i += 1
    indexes = np.random.permutation(indexes)
    others = np.array(others)
    train = np.concatenate((data[indexes[0:7000]], data[others]))
    target = np.concatenate((data_target[indexes[0:7000]], data_target[others]))

    nan_cols = [2]
    hashed_cols = []
    normalize_nan(train, nan_cols)
    normalize_hashes(train, hashed_cols)
    scaler = preprocessing.Scaler()
    train = scaler.fit_transform(train)

    test = data[indexes[7000:]]
    test_target = data_target[indexes[7000:]]
    normalize_nan(test, nan_cols)
    normalize_hashes(test, hashed_cols)
    test = scaler.transform(test)

    n_features = train.shape[1]
    print 'Post pre-processing features:', n_features

    # Some good default values for classification
    clf = RandomForestClassifier(n_estimators=100,
                                 max_features='sqrt',
                                 max_depth=None, n_jobs=-1)

    print 'Starting Cross-Validation training and evaluation'
    cvround = 1
    cvrounds = 5
    cv = cross_validation.KFold(len(train), k=cvrounds, indices=False)
    results = np.array([], dtype='uint8')
    for traincv, testcv in cv:
        print 'Starting CV round:', cvround, 'of', cvrounds
        pclass = clf.fit(train[traincv], target[traincv]).predict(train[testcv])
        results = np.concatenate((results, pclass))
        cvround += 1
    evaluation = evaluate(target, results)
    print 'Estimated CV accuracy:', evaluation
    print 'Total dead:', sum(target == 0)
    print 'Total just alive:', sum(target == 1)
    print 'Total acquired:', sum(target == 2)
    print 'Total in IPO:', sum(target == 3)

    print 'Predictions on the alive:'
    clf.fit(train, target)
    results = clf.predict(test)
    print 'Predicted dying:', sum(np.logical_and(test_target == 1, results == 0))
    print 'Predicted being acquired:', sum(np.logical_and(test_target == 1, results == 2))
    print 'Predicted entering IPO:', sum(np.logical_and(test_target == 1, results == 3))

if __name__ == "__main__":
    main()
