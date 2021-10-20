from pylab import *
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from hmmlearn import hmm

def hmm_spec_2c(X, Y):

    # print("Monitoring asthma medication adherence through content based audio classification")

    # print("Lalos et al. MFCC's, two-class problem")
    # t_beg = time.time()

    # print("Select features for input")

    # X = loadtxt('mfcc_.csv', delimiter=' ')
    # NewX_train, NewX_test, Newy_train, Newy_test = train_test_split(X, Y, test_size=0.3, shuffle=True)
    # X_train = pd.DataFrame(NewX_train)

    # print("Select 2 class or 4 class problem")
    '''
    questions = [
        inquirer.List('classes',
                      message='Which type of problem prefer?',
                      choices=['2-class problem (drug actuation or other sound', '4-class problem (inhalation, '
                                                                                 'exhalation, actuation, noise)'],
                      ),
    ]
    answers = inquirer.prompt(questions)
    print('nikos', answers['classes'])
    if answers['classes'] == '2-class problem (drug actuation or other sound':
    '''
    if True:
        print('EEEEEEEEE')
        clf = hmm.GaussianHMM(n_components=2,
                        covariance_type="full",
                        init_params="cm", params="cm", n_iter=100)

        Y_ = class_2Y(Y)
        y_pred = cross_val_predict(clf, X, cv=10)
        conf_mat = confusion_matrix(Y_, y_pred)
        print(conf_mat)
        # print(y_pred['test_score'])
        # print(y_pred['test_score'].mean())
        cm = conf_mat.astype('float') / conf_mat.sum(axis=1)[:, np.newaxis]
        print(cm)
        #t0 = time.time()
        #print('time elapsed for the algorithm to train and test: ', t0 - t_beg)
        print("Accuracy score: ", accuracy_score(Y_, y_pred))

        # Classification report
        targets = ['Class-0', 'Class-1']
        print('\n', classification_report(Y_, y_pred, target_names=targets))


def hmm_spec_4c(X, Y):

    # print("Monitoring asthma medication adherence through content based audio classification")

    # print("Lalos et al. MFCC's, two-class problem")
    # t_beg = time.time()

    # print("Select features for input")

    # X = loadtxt('mfcc_.csv', delimiter=' ')
    # NewX_train, NewX_test, Newy_train, Newy_test = train_test_split(X, Y, test_size=0.3, shuffle=True)
    # X_train = pd.DataFrame(NewX_train)

    # print("Select 2 class or 4 class problem")
    '''
    questions = [
        inquirer.List('classes',
                      message='Which type of problem prefer?',
                      choices=['2-class problem (drug actuation or other sound', '4-class problem (inhalation, '
                                                                                 'exhalation, actuation, noise)'],
                      ),
    ]
    answers = inquirer.prompt(questions)
    print('nikos', answers['classes'])
    if answers['classes'] == '2-class problem (drug actuation or other sound':
    '''
    if True:
        print('EEEEEEEEE')
        clf = hmm.GMMHMM(n_components=4,
                         covariance_type="diag",
                         init_params="cm", params="cm")

        # Y_ = class_2Y(Y)


        Y_ = class_4Y(Y)


        y_pred = cross_val_predict(clf, X, Y_.values.ravel(), cv=10)
        conf_mat = confusion_matrix(Y_, y_pred)
        print(conf_mat)
        # print(y_pred['test_score'])
        # print(y_pred['test_score'].mean())
        cm = conf_mat.astype('float') / conf_mat.sum(axis=1)[:, np.newaxis]
        print(cm)
        #t0 = time.time()
        #print('time elapsed for the algorithm to train and test: ', t0 - t_beg)
        print("Accuracy score: ", accuracy_score(Y_, y_pred))

        # Classification report
        targets = ['Class-0', 'Class-1', 'Class-2', 'Class-3']
        print('\n', classification_report(Y_, y_pred, target_names=targets))












