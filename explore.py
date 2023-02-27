import toml
import glob
import os
import librosa
import matplotlib
matplotlib.use('TkAgg',force=True)
from matplotlib import pyplot as plt
from librosa import display
from playsound import playsound

def getAudioFiles(path):
    files = {}
    classes = glob.glob(path+"/*")
    for c in classes:
        label = os.path.split(c)[-1]
        if label != "shards":
            labelFiles = glob.glob(c+"/*")
            files.update({label:labelFiles})
    return files

def displayWaveform(file_path):
    raw_audio = librosa.load(file_path)
    plt.figure()
    display.waveshow(raw_audio[0], sr=raw_audio[1])

def displaySpect(file_path, n_fft=2048, hop_length=None, win_length=None, window="hann", center=True, dtype=None,
                 pad_mode="reflect"):
    raw_audio = librosa.load(file_path)
    spect = librosa.stft(raw_audio[0], n_fft=n_fft, hop_length=hop_length, win_length=win_length, window=window,
                         center=center, dtype=dtype, pad_mode=pad_mode)
    plt.figure()
    display.specshow(spect, sr=raw_audio[1])

def play(file_path):
    playsound(file_path)

def examine(file_path):
    print(f"Examining {os.path.split(file_path)[-1]}...")
    print("Displaying waveform...")
    displayWaveform(file_path)
    print("Displaying spectrogram...")
    displaySpect(file_path)
    print("Playing audio...")
    play(file_path)

if __name__ == "__main__":
    params_file = "params.toml"
    params = toml.load(params_file)
    dataset_dir = params['data']['dataset_dir']
    dataset_name = params['data']['dataset_name']
    data_path = os.path.join(dataset_dir, dataset_name)

    sounds = getAudioFiles(data_path)['idling']

    examine(sounds[0])


