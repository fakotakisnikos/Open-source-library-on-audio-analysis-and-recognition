import keras
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
import keras.optimizers
import keras.losses

def baseline_model_cnn_2016(num_of_classes):


    model = Sequential()
    model.add(Conv1D(filters=16, kernel_size=100, activation='elu', input_shape=(200, 20)))
    model.add(Conv1D(filters=16, kernel_size=100, activation='elu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())
    model.add(Dense(256, activation='elu'))
    model.add(Dropout(0.5))
    model.add(Dense(16, activation='elu'))
    # model.add(Dropout(0.5))
    model.add(Dense(num_of_classes, activation='softmax'))
    opt = SGD(learning_rate=0.1, momentum=0.9)
    model.compile(loss=keras.losses.categorical_crossentropy, optimizer=opt, metrics=['accuracy'])

    return model

