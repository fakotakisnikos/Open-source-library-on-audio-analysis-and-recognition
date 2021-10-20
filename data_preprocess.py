from os.path import join, abspath
from os import getcwd
import os
import numpy as np
from tensorflow.keras.utils import to_categorical
from glob import glob
import wave
from scipy import io
import scipy.io.wavfile
from feature_extraction import *

def data_collection_v01():

    np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})

    classes = ['Exhale', 'Inhale', 'Drug', 'Noise']

    path = abspath(getcwd())

    _f1 = join(abspath(getcwd()), "tab_database/_f1/")
    _g1 = join(abspath(getcwd()), "tab_database/_g1/")
    _a1 = join(abspath(getcwd()), "tab_database/_a1/")

    f1 = glob(os.path.join("tab_database/_f1/", "*wav"))
    g1 = glob(os.path.join("tab_database/_g1/", "*wav"))
    a1 = glob(os.path.join("tab_database/_a1/", "*wav"))

    filenames = f1
    filenames.extend(g1)
    filenames.extend(a1)

    annotation_f1 = np.genfromtxt(_f1 + 'annotation_f1_dei.csv', delimiter=',', dtype='<U50')
    annotation_g1 = np.genfromtxt(_g1 + 'annotation_g1_dei.csv', delimiter=',', dtype='<U50')
    annotation_a1 = np.genfromtxt(_a1 + 'annotation_a1_dei.csv', delimiter=',', dtype='<U50')

    annotation = np.concatenate((annotation_f1, annotation_g1, annotation_a1))

    Y = []  # classes for train dataset
    X = []  # training dataset

    for file_element in filenames:
        w = scipy.io.wavfile.read(file_element)
        w_ = wave.open(file_element)
        maxval = 2 ** (w_.getsampwidth() * 8 - 1)
        fs = w[0]
        audioData = w[1]

        audioData = audioData / maxval
        # audioData = (audioData + 1) / 2

        head_tail = os.path.split(file_element)
        A = annotation[annotation[:, 0] == head_tail[1], :]

        for row in A:
            # label
            mLabel = row[1]
            startA = int(row[2])  # sample from which the respiratory phase begins
            stopA = int(row[3])  # sample at which the respiratory phase stops
            for i in range(startA, (stopA - 4000),
                           500):
                # loop to slide a window of size 4000 with step of 500 samples.
                # There is an overlapping of 3500 samples.
                c = classes.index(mLabel)
                # label
                Y.append(c)
                dat = np.asarray(audioData[i:i + 4000])
                # dat.resize((1, 250, 16))
                # data
                X.append(dat)
    data_v = np.asarray(X)
    savetxt('dataX_v01.csv', data_v, delimiter=' ')
    print("Data X shape: ", data_v.shape)  # (1161, 40)
    label_v = to_categorical(Y, 4)
    savetxt('labelX_v01.csv', label_v, delimiter=' ')
    print("Label Y shape: ", label_v.shape)  # (1161, 40)
    # data_v.resize(len(data_v), 250, 16, 1)
    return data_v, label_v


def create_features():
    X = loadtxt('dataX_v01.csv', delimiter=' ')
    create_spectrograms(X)
    create_cepstrum(X)

    create_mfcc(X)