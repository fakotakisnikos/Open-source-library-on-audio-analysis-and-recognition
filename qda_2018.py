from characteristics_v01 import *


def qda_v02_2_class(X, Y):
    clf = QDA()
    Y_ = class_2y(Y)
    print(X.shape, Y_.shape)
    clf.fit(X, Y_)
    print(clf.score(X, Y_))
    print(clf.predict([X[0]]))


def lda_v02_2_class(X, Y):
    clf = LDA()
    Y_ = class_2y(Y)
    print(X.shape, Y_.shape)
    clf.fit(X, Y_)
    print(clf.score(X, Y_))
    print(clf.predict([X[0]]))

