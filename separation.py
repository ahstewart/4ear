import pyroomacoustics as pra
import librosa
import numpy as np
from scipy.io import wavfile

aud_path = "C:\\Users\\ahste\\OneDrive\\ML Datasets\\AudioSet\\datasets\\cars_or_not\\car\\0cr9YtzgZT4.mp3"

aud, sr = librosa.load(aud_path)
#aud = np.array([aud], dtype=np.float32)
#aud = aud.T

fft_size = 4096
hop = fft_size // 2
aud_stft = pra.transform.analysis(aud, fft_size, hop, win=pra.hann(fft_size))
a = np.array([aud_stft])


#sep = pra.bss.auxiva(a)