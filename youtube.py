# -*- coding: utf-8 -*-
"""
Created on Sun Nov  7 20:21:30 2021

@author: ahste
"""

import librosa
from pytube import YouTube
import os
import datetime

youtube_url_prefix = "http://youtube.com/watch?v="
cwd = os.getcwd()
temp_filename = "temp.mp4"


def get_aud(df_row, sample_rate, dur, id_col='YTID', start_col=' start_seconds', 
            end_col=' end_seconds', stream_index=0, mono=True):
    youtube_id = df_row[id_col]
    url = youtube_url_prefix + youtube_id
    vid = YouTube(url)
    audio_stream = vid.streams.filter(only_audio=True, file_extension='mp4').first()
    out_file = audio_stream.download(filename=temp_filename)
    base, ext = os.path.splitext(out_file)
    start = str(datetime.timedelta(seconds=(df_row[start_col])))
    duration = str(datetime.timedelta(seconds=(df_row[end_col]-df_row[start_col])))
    wav_file = base + ".wav"
    os.system(f"ffmpeg -i \"{out_file}\" -ss {start} -t {duration} \"{wav_file}\"")
    audio = librosa.load(wav_file, sr=sample_rate, duration=dur, mono=True)
    os.remove(out_file)
    os.remove(wav_file)
    return audio


def make_spect(aud, n_fft, hop_length, win_length, win_func, center, pad_mode):
    return librosa.stft(aud, n_fft=n_fft, hop_length=hop_length, 
                        win_length=win_length, window=win_func, center=center,
                        pad_mode=pad_mode)


def download_waveform(df_row, path, id_col='YTID', start_col=' start_seconds',
                      end_col=' end_seconds', stream_index=0, file_format='mp3'):
    youtube_id = df_row[id_col]
    url = youtube_url_prefix + youtube_id
    fail_count = 0
    try:
        vid = YouTube(url)
        audio_stream = vid.streams.filter(only_audio=True, file_extension='mp4').first()
        out_file = audio_stream.download(filename=os.path.join(path, youtube_id))
        base, ext = os.path.splitext(out_file)
        start = str(datetime.timedelta(seconds=(df_row[start_col])))
        duration = str(datetime.timedelta(seconds=(df_row[end_col]-df_row[start_col])))
        final_file = base + f".{file_format}"
        os.system(f"ffmpeg -i \"{out_file}\" -ss {start} -t {duration} \"{final_file}\" -hide_banner -loglevel error")
        os.remove(out_file)
    except Exception as e:
        # print("\n\tNo matching YouTube clip found... Skipping record")
        # print("\n\tVideo with YouTube ID " + str(e))
        pass


def create_waveform_ds(df, download_func, path, id_col='YTID', start_col=' start_seconds', 
                 end_col=' end_seconds', stream_index=0):
    df.apply(lambda row: download_func(df_row=row, path=path, id_col=id_col, start_col=start_col,
                                       end_col=end_col, stream_index=stream_index),
                                       axis=1)
