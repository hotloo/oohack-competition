#!/bin/python2
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


def evaluate(target, predicted):
    return sum(target == predicted)


def main():
    #Skipping the header row with [1:]
    print 'Loading train data'
    # dataset = np.genfromtxt(open('../data/train.csv', 'r'), delimiter=',',
    #                         dtype='Float64', skip_header=0)
    dataset = np.load('../data/train.npy')
    target = np.array([x[0] for x in dataset])
    train = np.array([x[1:] for x in dataset])
    # target = np.load('../data/target.npy')
    # train = np.load('../data/train.npy')

    print 'Data loaded, starting pre-processing'
    scaler = preprocessing.Scaler()
    scaler.fit(train)
    train = scaler.transform(train)

    #train = np.array([normalize(x) for x in train])
    #PCA, keep 95% of variance WRONG IF TRAIN/TEST ARE NOT EQUALLY NORMALIZED!
    #Also tricky if the proportion of all numbers is not the same
    # pca = decomposition.PCA()
    # pca.fit(train)
    # ratio = 0
    # comp = 0
    # for i in pca.explained_variance_ratio_:
    #     ratio += i
    #     if ratio > 0.95:
    #         break
    #     comp += 1
    # pca.n_components = comp
    # train = pca.fit_transform(train)

    n_features = train.shape[1]
    print 'Post pre-processing features:', n_features

    # Some good default values for classification
    clf = RandomForestClassifier(n_estimators=100,
                                 max_features='sqrt',
                                 max_depth=None, n_jobs=-1)

    # Some good default values for classification
    #clf = svm.SVC(gamma=0.001, kernel='poly', degree=3)
    #clf = svm.SVC(gamma=0.001, kernel='linear')
    #clf = LinearSVC(penalty='l1', dual=False, tol=1e-3)
    #clf = LinearSVC(loss='l2', penalty='l1', dual=False, tol=1e-3)
    #clf = LinearSVC(loss='l2', penalty='l2', dual=False, tol=1e-3)

    #clf = KNeighborsClassifier(n_neighbors=10, warn_on_equidistant=False)
    #clf = linear_model.LogisticRegression()
    #clf = MultinomialNB(alpha=.01)

    cv = cross_validation.KFold(len(train), k=5, indices=False)

    results = np.array([], dtype='uint8')
    print 'Starting Cross-Validation training and evaluation'
    for traincv, testcv in cv:
        pclass = clf.fit(train[traincv], target[traincv]).predict(train[testcv])
        results = np.concatenate((results, pclass))
    print 'Estimated (CV) accuracy: ' + str(evaluate(target, results) / len(target))

    # print 'Loading test data'
    # # test = np.genfromtxt(open('../data/test.csv', 'r'), delimiter=',',
    # #                      dtype='Float64', skip_header=0)
    # test = np.load('../data/test.npy')
    # print 'Data loaded, starting pre-processing'
    # #test = np.array([normalize(x) for x in test])
    # #Same scaling
    # test = scaler.transform(test)
    # #Same PCA transformation
    # # test = pca.fit_transform(test)

    # print 'Retraining with everything and saving results'
    # pclass = clf.fit(train, target).predict(test)
    # np.savetxt('../data/result.csv', pclass, delimiter=',', fmt='%d')

if __name__ == "__main__":
    main()
