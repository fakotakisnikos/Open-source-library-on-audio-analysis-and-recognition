import inquirer
from sklearn import svm
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis as QDA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.mixture import GaussianMixture
from cnn_2016 import *
from cnn_2020 import *
from classification_report import *
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
import pandas as pd
from hmmlearn import hmm
from keras.wrappers.scikit_learn import KerasClassifier


def technique_selection():
    #X = loadtxt('dataX_v01.csv', delimiter=' ')
    #Y = loadtxt('labelX_v01.csv', delimiter=' ')


    questions = [
        inquirer.Confirm('continue',
                         message="Should I continue (y/n):"),
    ]
    answers = inquirer.prompt(questions)
    if answers['continue'] == 'N':
        exit(0)

    questions = [
        inquirer.List('method',
                      message="Which technique do you want to use?",
                      choices=['Random Forests', 'SVM', 'AdaBoost', 'Convolutional NN (2016)',
                               'LSTM', 'Gaussian MM', 'Decision Tree (2014)', 'Linear Discriminant Analysis',
                               'Quadratic Discriminant Analysis', 'Hidden Markov Model', 'Convolutional NN (2020)'],
                      ),
    ]
    method = inquirer.prompt(questions)
    print("nikos", method['method'])

    if method['method'] == 'Random Forests':
        method_num = 0
    elif method['method'] == 'SVM':
        method_num = 1
    elif method['method'] == 'AdaBoost':
        method_num = 2
    elif method['method'] == 'Convolutional NN (2016)':
        method_num = 3
    elif method['method'] == 'LSTM':
        method_num = 4
    elif method['method'] == 'Gaussian MM':
        method_num = 5
    elif method['method'] == 'Decision Tree (2014)':
        method_num = 6
    elif method['method'] == 'Linear Discriminant Analysis':
        method_num = 7
    elif method['method'] == 'Quadratic Discriminant Analysis':
        method_num = 8
    elif method['method'] == 'Hidden Markov Model':
        method_num = 9
    elif method['method'] == 'Convolutional NN (2020)':
        method_num = 10
    return method_num


def classes_selection():

    questions = [
        inquirer.List('nclss',
                      message="Do you want a two class problem or a four class problem ?",
                      choices=['Two-class problem', 'Four-class problem'],
                      ),
    ]
    clss = inquirer.prompt(questions)
    print(clss['nclss'])

    if clss['nclss'] == 'Two-class problem':
        cls_num = 2
    elif clss['nclss'] == 'Four-class problem':
        cls_num = 4

    return cls_num


def feature_selection_v02():
    questions = [
        inquirer.List('feat',
                      message="Which type of feature do you want to use?",
                      choices=['Time domain', 'Spectrogram', 'MFCCs', 'Cepstrum'],
                      ),
    ]
    feature_ex = inquirer.prompt(questions)

    if feature_ex['feat'] == 'Time domain':
        feat_num = 0
    elif feature_ex['feat'] == 'Spectrogram':
        feat_num = 1
    elif feature_ex['feat'] == 'MFCCs':
        feat_num = 2
    elif feature_ex['feat'] == 'Cepstrum':
        feat_num = 3

    return feat_num

    execute_func(method_num, cls_num, feat_num)


def class_2y(Y):
    Y_new = np.empty((0,), int)
    for i in range(len(Y)):
        if Y[i, 2] == 1.000:
            Y_new = np.append(Y_new, 1)
        else:
            Y_new = np.append(Y_new, 0)

    Y_ = pd.DataFrame(Y_new)
    return Y_


def class_4y(Y):
    Y_new = np.empty((0,), int)
    for i in range(len(Y)):
        if Y[i, 0] == 1.000:
            Y_new = np.append(Y_new, 0)
        elif Y[i, 1] == 1.000:
            Y_new = np.append(Y_new, 1)
        elif Y[i, 2] == 1.000:
            Y_new = np.append(Y_new, 2)
        elif Y[i, 3] == 1.000:
            Y_new = np.append(Y_new, 3)

    Y_ = pd.DataFrame(Y_new)
    return Y_


def create_labels(num):
    Y = loadtxt('labelX_v01.csv', delimiter=' ')
    if num == 2:
        Y_ = class_2y(Y)
    elif num == 4:
        Y_ = class_4y(Y)
    return Y_


def feature_selection(num):
    X = loadtxt('dataX_v01.csv', delimiter=' ')
    if num == 0:
        input = X
    elif num == 1:
        input = loadtxt('spectrogram_v01.csv', delimiter=' ')
    elif num == 2:
        input = loadtxt('mfcc_v01.csv', delimiter=' ')
    elif num == 3:
        input = loadtxt('cepstrum_v01.csv', delimiter=' ')
    return input


def main_selection():

    num = technique_selection()
    feat = feature_selection_v02()
    cls = classes_selection()

    if num == 2:
        X = feature_selection(feat)
        Y_ = create_labels(cls)
        model = AdaBoostClassifier(DecisionTreeClassifier(max_depth=2), n_estimators=600, learning_rate=1)
        classification_report(X, Y_, model)

    elif num == 0:
        X = feature_selection(feat)
        Y_ = create_labels(cls)
        model = RandomForestClassifier(
            n_estimators=100,
            criterion='entropy',
            warm_start=True,
            max_features='sqrt',
            oob_score='True',  # more on this below
            random_state=69
        )
        classification_report(X, Y_, model)

    elif num == 1:
        X = feature_selection(feat)
        Y_ = create_labels(cls)
        model = svm.SVC(decision_function_shape='ovr')
        classification_report(X, Y_, model)

    elif num == 3:
        X = feature_selection(feat)
        Y_ = create_labels(cls)
        model = QDA()
        classification_report(X, Y_, model)

    elif num == 4:
        X = feature_selection(feat)
        Y_ = create_labels(cls)
        model = LDA()
        classification_report(X, Y_, model)

    elif num == 5:
        X = feature_selection(feat)
        Y_ = create_labels(cls)
        model = GaussianMixture(n_components=2)
        classification_report(X, Y_, model)

    elif num == 6:
        X = feature_selection(feat)
        Y_ = create_labels(cls)
        model = hmm.GaussianHMM(n_components=2,
                        covariance_type="full",
                        init_params="cm", params="cm", n_iter=100)
        classification_report(X, Y_, model)

    elif num == 3:
        X = feature_selection(feat)
        X = np.reshape(X, (12687, 200, 20))
        Y_ = create_labels(cls)
        model = KerasClassifier(build_fn=baseline_model_cnn_2016(cls), epochs=50, verbose=1)
        classification_report(X, Y_, model)

    elif num == 10:
        X = feature_selection(feat)
        X = np.reshape(X, (12687, 250, 16, 1))
        Y_ = create_labels(cls)
        model = KerasClassifier(build_fn=baseline_model_cnn_2020(cls), epochs=50, verbose=1)
        classification_report(X, Y_, model)

    elif num == 9:
        X = feature_selection(feat)
        Y_ = create_labels(cls)
        model = hmm.GaussianHMM(n_components=2,
                        covariance_type="full",
                        init_params="cm", params="cm", n_iter=100)
        classification_report(X, Y_, model)

    return model



