import numpy as np
import pywt
from sklearn.metrics import confusion_matrix


def decision_t_2014(X):
    Y_t = []
    Y_p = []
    for i in range(len(X)):
        for j in range(0, len(X[0]) - 1000, 1000):
            step = 100
            # print("nikos")
            coef_mat = []
            wavelet = 'cmor1.5-1.0'
            sst = X[i, j:j+1000]
            for i in range(0, np.shape(sst)[0], step):
                w = sst[i: i + step]
                coef, freqs = pywt.cwt(w, np.arange(1.625, 2.00, 0.005), 'morl')
                coef_mat.append(coef)
            cwtmat = np.array(coef_mat)
            frequencies = pywt.scale2frequency('morl', np.arange(1.625, 2.00, 0.005)) * 8000

            # print(cwtmat.shape) # (10, 75, 100)

            thres = 50  # .1*(10**223)

            # print("Frequencies: ", frequencies)
            # print("coefmatshape:", coef.shape)  # (75, 1000) (scale, samples = 8000 * time)
            # print(cwtmat)

            first = np.zeros(np.shape(cwtmat)[0])
            second = np.zeros(np.shape(cwtmat)[0])
            cwt_window = np.zeros(np.shape(cwtmat)[0])
            for k in range(np.shape(cwtmat)[0]):
                cwt_window[k] = np.sum(cwtmat[k, :, :] ** 2)

            '''
            print(cwt_window.shape)
            m = 0
            ccwt = 0
            for i in range(np.shape(cwtmat)[2]):
                ccwt += cwt_window[i]
                m+=1
            ccwt = ccwt/m
            print("cwt: ", ccwt, "Y[2]: ", Y[2])
            '''

            for i in range(np.shape(cwt_window)[0]):
                # print(cwt_window[i])
                if (cwt_window[i] > thres):
                    # print("(first threshold) Potential blister sound, at: ", i, " window, with CWT squared coefficients: ", cwt_window[i])
                    first[i] = 1
                    if i < np.shape(cwt_window)[0] - 4:
                        if cwt_window[i] * 0.75 > cwt_window[i + 4]:
                            second[i] = 1
                    elif i - 4 >= 0:
                        if cwt_window[i] * 0.75 > cwt_window[i - 4]:
                            second[i] = 1
                    else:
                        # 56.0 msec
                        if (cwt_window[i - 4] < cwt_window[i] * 0.75) and (cwt_window[i + 4] < cwt_window[i] * 0.75):
                            second[i] = 1

            y_true = []
            y_pred = []
            for i in range(np.shape(cwt_window)[0]):
                # print(Y[2])
                if first[i] == 1 and second[i] == 1:
                    if Y[2] == 1.000:
                        y_true = np.append(y_true, 1)
                        y_pred = np.append(y_pred, 1)
                        break
                    else:
                        y_true = np.append(y_true, 0)
                        y_pred = np.append(y_pred, 1)
                        break
                else:
                    if Y[2] == 1.000:
                        continue
                        # y_true = np.append(y_true, 1)
                        # y_pred = np.append(y_pred, 0)
                    else:
                        y_true = np.append(y_true, 0)
                        y_pred = np.append(y_pred, 0)
                        break

            Y_t = np.append(Y_t, y_true)
            Y_p = np.append(Y_p, y_pred)

    cm1 = confusion_matrix(Y_t, Y_p)
    print('Confusion Matrix : \n', cm1)

    total1 = sum(sum(cm1))
    #####from confusion matrix calculate accuracy
    accuracy1 = (cm1[0, 0] + cm1[1, 1]) / total1
    print('Accuracy : ', accuracy1)

    sensitivity1 = cm1[0, 0] / (cm1[0, 0] + cm1[0, 1])
    print('Sensitivity : ', sensitivity1)

    specificity1 = cm1[1, 1] / (cm1[1, 0] + cm1[1, 1])
    print('Specificity : ', specificity1)