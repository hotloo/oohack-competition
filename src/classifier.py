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
    # tp = sum(np.logical_and(target, predicted))
    # fp = sum(np.logical_and(target == 0, predicted == 1))
    # fn = sum(np.logical_and(target == 1, predicted == 0))
    # print total, correct, tp, fp, fn
    accuracy = correct / total
    # precision = tp / (tp + fp)
    # recall = tp / (tp + fn)
    # f1 = 2 * precision * recall / (precision + recall)
    # return (accuracy, recall, precision, f1)
    return accuracy


def main():
    print 'Loading train data'
    dataset = np.load('../data/oohack_train.npy')
    print 'Loading test data'
    # testset = np.load('../data/oohack_test.npy')

    target = np.array([x[0] for x in dataset])
    train = np.array([x[1:] for x in dataset])
    # test_target = np.array([x[0] for x in testset])
    # test = np.array([x[1:] for x in testset])

    print 'Data loaded, starting pre-processing'
    nan_cols = [2]
    hashed_cols = []
    normalize_nan(train, nan_cols)
    normalize_hashes(train, hashed_cols)
    scaler = preprocessing.Scaler()
    train = scaler.fit_transform(train)

    # normalize_nan(test, nan_cols)
    # normalize_hashes(test, hashed_cols)
    # test = scaler.transform(test)

    n_features = train.shape[1]
    print 'Post pre-processing features:', n_features

    # Some good default values for classification
    clf = RandomForestClassifier(n_estimators=100,
                                 max_features='sqrt',
                                 max_depth=None, n_jobs=-1)

    # Some good default values for classification
    #clf = svm.SVC(cache_size=512, gamma=0.001, kernel='poly', degree=3,
                   #class_weight='auto')
    #clf = svm.SVC(cache_size=512, gamma=0.001, kernel='rbf', degree=3,
                   #class_weight='auto')
    #clf = svm.SVC(cache_size=512, gamma=0.001, kernel='linear',
                   #class_weight='auto')
    #clf = LinearSVC(penalty='l1', dual=False, tol=1e-3)
    #clf = LinearSVC(loss='l2', penalty='l1', dual=False, tol=1e-3)
    #clf = LinearSVC(loss='l2', penalty='l2', dual=False, tol=1e-3)

    #clf = KNeighborsClassifier(n_neighbors=10, warn_on_equidistant=False)
    #clf = linear_model.LogisticRegression()
    #clf = MultinomialNB(alpha=.01)

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
    print 'Estimated CV accuracy: ', evaluation
    # print 'Estimated CV accuracy: ', evaluation[0]
    # print 'Estimated CV precision: ', evaluation[1]
    # print 'Estimated CV recall: ', evaluation[2]
    # print 'Estimated CV F1 score: ', evaluation[3]


    # print 'Retraining with everything and saving results'
    # pclass = clf.fit(train, target).predict(test)
    # evaluation = evaluate(test_target, pclass)
    # print 'Generalization accuracy: ', evaluation[0]
    # print 'Generalization precision: ', evaluation[1]
    # print 'Generalization recall: ', evaluation[2]
    # print 'Generalization F1 score: ', evaluation[3]
    # np.savetxt('../data/result.csv', pclass, delimiter=',', fmt='%d')

if __name__ == "__main__":
    main()
