import keras
from tensorflow.keras.layers import *
from tensorflow.keras.models import load_model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
import keras.optimizers
import keras.losses
from pylab import *
from keras.models import model_from_json
import sounddevice as sd
import time


def baseline_model_cnn_2020(num_classes):
    ##CNN Architecture
    inhaler_model = Sequential()
    inhaler_model.add(Conv2D(16, kernel_size=(3, 3), activation='elu', input_shape=(250, 16, 1), padding='same',
                             use_bias=True))  # activation can be linear,relu,elu etc
    inhaler_model.add(MaxPooling2D((2, 2), padding='same'))
    inhaler_model.add(Dropout(0.2))
    inhaler_model.add(Conv2D(16, kernel_size=(5, 5), activation='elu', padding='same', use_bias=True))
    inhaler_model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
    inhaler_model.add(Dropout(0.1))
    # inhaler_model.add(Conv2D(16, kernel_size=(6, 6), activation='elu', padding='same', use_bias=True))
    # inhaler_model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
    inhaler_model.add(Flatten())
    inhaler_model.add(Dense(64, activation='elu'))
    inhaler_model.add(Dense(128, activation='elu'))
    inhaler_model.add(Dense(64, activation='elu'))
    inhaler_model.add(Dense(num_classes, activation='softmax'))
    inhaler_model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adam(),
                          metrics=['accuracy'])

    return inhaler_model


def save_model(X, Y, model):
    jmodel = model
    jmodel.fit(X, Y, epochs=50)
    # serialize model to JSON
    model_json = jmodel.to_json()
    with open("model_cnn_2020_4c_v03.json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    jmodel.save_weights("model_cnn_2020_4c_v03.h5")
    print("Saved model to disk")


def load_model():
    # load json and create model
    json_file = open('model_cnn_2020_4c_v03.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights("model_cnn_2020_4c_v03.h5")
    print("Loaded model from disk")
    return loaded_model


def live_audio_streaming_classification():
    fs = 8000
    duration = 0.5  # seconds
    model = load_model()
    time.sleep(1)
    print('Start recording')
    myrecording = sd.rec(int(duration * fs), samplerate=fs, channels=1)
    # print(myrecording.shape)
    # print(myrecording)
    X = np.reshape(myrecording, (1, 250, 16, 1))
    pr = model.predict(X)
    dtype = [('Type', np.unicode_, 16), ('Probability', float)]
    values = [('Exhale', pr[0][0]), ('Inhale', pr[0][1]), ('Drug', pr[0][2]), ('Noise', pr[0][3])]
    a = np.array(values, dtype=dtype)  # create a structured array
    print(np.sort(a, order='Probability')[::-1])




