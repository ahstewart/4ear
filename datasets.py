# -*- coding: utf-8 -*-
"""
Created on Mon Nov 15 16:55:10 2021

@author: ahste
"""

import parse_audioset
import youtube
import os
import sys
import tensorflow as tf
import tensorflow_io as tfio
import numpy as np

def create_dataset(dataset_dir, dataset_name, classes=[], dfs=[], rep="mp3"):
    # setup dataset/class directories
    dataset_path = os.path.join(dataset_dir, dataset_name)
    if os.path.exists(dataset_path) == False:
        os.mkdir(dataset_path)
    class_paths = [os.path.join(dataset_path, i) for i in classes]
    for path in class_paths:
        try:
            os.mkdir(path)
        except FileExistsError:
            print("All or some of the dataset/class pairs entered already exist.")
            sys.exit()
    # use each class's provided df to create dataset of the type "rep"
    # if rep == "mp3":
        
    
def dataset_dicts(df, classes=[], keywords=[], modes=[], label_col="labels"):
    if len(classes) == len(keywords) == len(modes):
        df_dict = {}
        for i in range(len(classes)):
            filtered_df = parse_audioset.filter_by_labels(df, labels=keywords[i],
                                                          mode=modes[i], 
                                                          label_col=label_col)
            df_dict[classes[i]] = filtered_df
        return df_dict
    else:
        print("Error: Length of classes, keywords, and modes lists must be equal")
        sys.exit()
        
def make_dataset(dataset_dir, dataset_name, dataset_dicts, id_col='YTID', 
                 start_col=' start_seconds', end_col=' end_seconds', stream_index=0):
    dataset_path = os.path.join(dataset_dir, dataset_name)
    if os.path.exists(dataset_path) == False:
        os.mkdir(dataset_path)
    for dsets in dataset_dicts:
        class_dir = os.path.join(dataset_path, dsets)
        if os.path.exists(class_dir) == False:
            os.mkdir(class_dir)
        dataset_dicts[dsets].apply(lambda row: youtube.download_waveform(row, class_dir,
                                                                         id_col=id_col,
                                                                         start_col=start_col,
                                                                         end_col=end_col,
                                                                         stream_index=stream_index),
                                   axis=1)

# def file_to_waveform(file_path):
#     audio = tfio.IOTensor.graph(tf.float32).from_audio(file_path)
#     aud_tensor = audio.to_tensor()
#     print(audio.shape[0])
#     if aud_tensor.shape[1] == 2:
#         print("it was 2!")
#         audio1 = aud_tensor[:, 0]
#         audio2 = aud_tensor[:, 1]
#         aud_tensor = (audio1 + audio2) / 2.0
#     elif aud_tensor.shape[1] == 1:
#         print("it was 1!")
#         aud_tensor = tf.squeeze(aud_tensor, axis=[-1])
#     aud_tensor = tf.cast(aud_tensor, np.float32)
#     label = tf.strings.split(file_path, os.sep)[-2]
#     return aud_tensor, label

def file_to_waveform(file_path):
    audio = tfio.audio.AudioIOTensor(file_path)
    aud_tensor = audio.to_tensor()
    print(audio.shape[0].numpy())
    if aud_tensor.shape[1] == 2:
        print("it was 2!")
        audio1 = aud_tensor[:, 0]
        audio2 = aud_tensor[:, 1]
        aud_tensor = (audio1 + audio2) / 2.0
    elif aud_tensor.shape[1] == 1:
        print("it was 1!")
        aud_tensor = tf.squeeze(aud_tensor, axis=[-1])
    aud_tensor = tf.cast(aud_tensor, np.float32)
    label = tf.strings.split(file_path, os.sep)[-2]
    return aud_tensor, label

def waveform_to_spectrogram(audio_tensor, nfft, window, stride):
    return tfio.audio.spectrogram(audio_tensor, nfft, window, stride), 

def file_to_waveform_py(file_path):
    audio_py = tf.py_function(file_to_waveform, [file_path], tf.float32)
    return audio_py
        

def load_dataset(dataset_dir, represent="waveform"):
    list_ds = tf.data.Dataset.list_files(str(dataset_dir+'/*/*'))
    tensor_labeled_ds = list_ds.map(file_to_waveform_py)
    if represent == "spectrogram":
        pass
    return tensor_labeled_ds, list_ds

    

