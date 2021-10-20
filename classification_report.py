from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix
import numpy as np

def classification_report(X, Y_, clf):
    y_pred = cross_val_predict(clf, X, Y_.values.ravel(), cv=10)
    conf_mat = confusion_matrix(Y_, y_pred)
    print(conf_mat)
    cm = conf_mat.astype('float') / conf_mat.sum(axis=1)[:, np.newaxis]
    print(cm)
    print("Accuracy score: ", accuracy_score(Y_, y_pred))
    targets = ['Class-0', 'Class-1', 'Class-2', 'Class-3']
    print('\n', classification_report(Y_, y_pred, target_names=targets))

