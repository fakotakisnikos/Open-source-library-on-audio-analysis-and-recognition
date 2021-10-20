import librosa
from pylab import *
import scipy
from scipy import signal


def create_spectrograms(X):
    feat = np.zeros((len(X), 129))
    for i in range(len(X)):
        freqs, times, spectrogram = scipy.signal.spectrogram(X[i, :])
        feat[i] = np.mean(spectrogram, axis=1)
    fea = np.asarray(feat)
    savetxt('spectrogram_v01.csv', fea, delimiter=' ')
    print("Spectrogram shape: ", fea.shape)  # (12687, 129)


def create_cepstrum(X):
    features = np.empty((0,), float)
    feat = np.zeros((len(X), 4000))
    for i in range(len(X)):
        signal = X[i, :]
        ceps = real_cepstrum(signal, None)
        feat[i] = ceps
    t_beg = time.time()

    fea = np.asarray(feat)

    savetxt('cepstrum_v01.csv', fea, delimiter=' ')
    print("Cepstrogram shape: ", fea.shape)  # (1161, 40)


def real_cepstrum(x, n=None):
    spectrum = np.fft.fft(x, n=n)
    ceps = np.fft.ifft(np.log(np.abs(spectrum))).real
    return ceps


def create_mfcc(X):
    feat = np.zeros((len(X), 8))
    for i in range(len(X)):
        # Mel-frequency cepstral coefficients (MFCCs)
        mfcc = librosa.feature.mfcc(y=X[i, :], sr=8000, n_mfcc=13)

        # temporal averaging
        feat[i] = np.mean(mfcc, axis=0)
    fea = np.asarray(feat)

    savetxt('mfcc_v01.csv', fea, delimiter=' ')
    print("MFCC shape: ", fea.shape)  # (12687, 8)

