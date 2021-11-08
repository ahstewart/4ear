# -*- coding: utf-8 -*-
"""
Created on Sun Nov  7 20:21:30 2021

@author: ahste
"""

import numpy as np
import librosa
import pandas as pd
from pytube import YouTube
import os
from pydub import AudioSegment

youtube_url = "http://youtube.com/watch?v="
cwd = os.getcwd()
temp_filename = "temp.mp4"

def get_aud(df_row, id_col="YTID", stream_index=0):
    youtube_id = df_row[id_col]
    url = youtube_url + youtube_id
    vid = YouTube(url)
    audio_stream = vid.streams.filter(only_audio=True, file_extension='mp4').first()
    out_file = audio_stream.download(filename=temp_filename)
    base, ext = os.path.splitext(out_file)
    wav_file = base + ".wav"
    wav_conversion = AudioSegment.from_file(out_file)
    wav_conversion.export(wav_file, format="wav")
    audio = librosa.load(wav_file)
    os.remove(out_file)
    os.remove(wav_file)
    return audio
    