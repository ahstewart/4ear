# -*- coding: utf-8 -*-
"""
Created on Mon Nov 15 16:55:10 2021

@author: ahste
"""

import warnings
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    import parse_audioset
    import youtube
    import os
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    import sys
    import tensorflow as tf
    # import tensorflow_io as tfio
    import numpy as np
    from tqdm import tqdm
    from multiprocessing import Pool, cpu_count
    import math
    import glob
    import shutil
    from sklearn import preprocessing, model_selection
    import librosa
    import random
    tqdm.pandas()


def _bytestring_feature(list_of_bytestrings):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=list_of_bytestrings))


def _bytestring_feature_value(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _int_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _float_feature(list_of_floats):  # float32
    return tf.train.Feature(float_list=tf.train.FloatList(value=list_of_floats))


def _float_feature_value(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def _parallelize(func, data):
    processes = cpu_count() - 1
    with Pool(processes) as pool:
        # We need the enclosing list statement to wait for the iterator to end
        # https://stackoverflow.com/a/45276885/1663506
        list(tqdm(pool.imap_unordered(func, data), total=len(data)))


class Dataset:
    
    def __init__(self, params_dict):
        print("Creating Dataset object...")
        self.params = params_dict
        self.dataset_loc = self.params['data']['dataset_dir']
        self.dataset_name = self.params['data']['dataset_name']
        self.dataset_dir = os.path.join(self.dataset_loc, self.dataset_name)
        self.dataset_mode = self.params['data']['origin']
        self.shard_name = self.params['data']['shard_name']
        self.shard_dir_name = self.params['data']['shard_dir_name']
        self.shard_loc = os.path.join(self.dataset_dir, self.shard_dir_name)
        self.train_cut = self.params['data']['train_cut']
        self.val_cut = self.params['data']['valid_cut']
        self.test_cut = self.params['data']['test_cut']
        self.sample_rate = self.params['data']['sample_rate']
        self.duration = self.params['data']['duration']
        self.n_data_points = int(self.sample_rate * self.duration)
        self.n_channels = 1
        self.final_format = self.params['data']['final_format']
        if self.dataset_mode == "audioset_csv":
            self.keywords = self.params['data']['audioset_csv']['pull_keywords']
            self.modes = self.params['data']['audioset_csv']['pull_modes']
            self.max_items = self.params['data']['audioset_csv']['max_items']
            self.csv_loc = self.params['data']['audioset_csv']['csv_loc']
            self.ont_loc = self.params['data']['audioset_csv']['ont_loc']
            self.csv_head = self.params['data']['audioset_csv']['csv_head']
            self.label_ix_start = self.params['data']['audioset_csv']['label_ix_start']
            self.label_ix_end = self.params['data']['audioset_csv']['label_ix_end']
            self.label_col = self.params['data']['audioset_csv']['label_column']
            self.classes = self.params['data']['audioset_csv']['classes']
            self.file_format = self.params['data']['audioset_csv']['format']
            self.id_col = self.params['data']['audioset_csv']['id_column']
            self.start_col = self.params['data']['audioset_csv']['start_column']
            self.end_col = self.params['data']['audioset_csv']['end_column']
            self.df = self._get_dataframe()
            self.df_dicts = self._get_dataset_dicts(df=self.df, classes=self.classes,
                                                    keywords=self.keywords,
                                                    modes=self.modes,
                                                    max_items=self.max_items,
                                                    label_col=self.label_col)
            self._make_dataset(self.dataset_loc, self.dataset_name, self.df_dicts,
                               self.id_col, self.start_col, self.end_col)
            self.max_shard_size = self.params['data']['files']['max_shard_size']
            self.compression_scaling_factor = self.params['data']['files']['compression_scaling_factor']
        elif self.dataset_mode == "files":
            self.max_shard_size = self.params['data']['files']['max_shard_size']
            self.compression_scaling_factor = self.params['data']['files']['compression_scaling_factor']
        if (self.final_format == "spect") or (self.final_format == "mel_spect"):
            self.nfft = self.params['data']['spect']['n_fft']
            self.hop_length = self.params['data']['spect']['hop_length']
            self.win_length = self.params['data']['spect']['win_length']
            self.window = self.params['data']['spect']['win_func']
            self.center = self.params['data']['spect']['center']
            self.spect_dtype = None
            self.pad_mode = self.params['data']['spect']['pad_mode']
            self.power = self.params['data']['spect']['power']
        self._get_classes()
        self.label_encoder = preprocessing.LabelEncoder()
        self.label_encoder.fit(self.classes)
        self._get_filenames()
        # Get average file size of 10% of randomly selected files in dataset
        filesizes = []
        shuffled_files = self.ds_filenames.copy()
        random.shuffle(shuffled_files)
        for i in range(self.n_files // 10):
            filesizes.append(os.path.getsize(shuffled_files[i]))
        self.filesize = np.mean(filesizes)
        # for some data pulls, no file size may be found. in this case calculate the file size based on
        # the file format and duration defined in the config file
        if np.isnan(self.filesize):
            if self.file_format == "mp3" or self.file_format == "ogg":
                bps = 128 * 10**3
                Bps = bps / 8
                self.filesize = Bps * self.duration
            elif self.file_format == "mp4":
                bps = 256 * 10**3
                Bps = bps / 8
                self.filesize = Bps * self.duration
            elif self.file_format == "wav" or self.file_format == "flac":
                bps = 1411 * 10**3
                Bps = bps / 8
                self.filesize = Bps * self.duration
        if (self.final_format == "spect") or (self.final_format == "mel_spect"):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                test_audio, s_rate = librosa.load(self.ds_filenames[0],
                                                  sr=self.sample_rate,
                                                  mono=True,
                                                  duration=self.duration)
            test_spect = self._get_spectrogram(test_audio, n_fft=self.nfft,
                                               hop_length=self.hop_length,
                                               win_length=self.win_length,
                                               window=self.window,
                                               center=self.center,
                                               dtype=self.spect_dtype,
                                               pad_mode=self.pad_mode)
            self.spect_shape = (test_spect.shape[0], test_spect.shape[1], self.n_channels)
            del test_audio, s_rate, test_spect
        self.waveform_shape = (self.sample_rate * self.duration, self.n_channels)
        del filesizes, shuffled_files
        self.batch_size = self.params['data']['batch_size']
        self.val_test_batch_size = self.params['data']['val_test_batch_size']
        self.n_epochs = self.params['data']['n_epochs']
        self.shuffle_yn = self.params['data']['shuffle_yn']
        if self.n_files > 5000:
            self.buffer_size = self.n_files // 5
        else:
            self.buffer_size = self.n_files
        self.random_seed = self.params['data']['random_seed']
        if self.dataset_mode == "audioset_csv" or self.dataset_mode == "files":
            self.convert()
        self.ds = self.read_shards()
        self.dd = self.prep_data(self.ds)
        
    def _get_dataframe(self):
        return parse_audioset.parse_csv(self.csv_loc, self.ont_loc, 
                                        csv_head=self.csv_head, 
                                        label_ix_start=self.label_ix_start,
                                        label_ix_end=self.label_ix_end,
                                        new_col_name=self.label_col)
        
    def _get_dataset_dicts(self, df, classes=[], keywords=[], modes=[], max_items=[], label_col="labels"):
        if len(classes) == len(keywords) == len(modes):
            df_dict = {}
            for i in range(len(classes)):
                filtered_df = parse_audioset.filter_by_labels(df, labels=keywords[i],
                                                              mode=modes[i],
                                                              max_items=max_items[i],
                                                              label_col=label_col)
                df_dict[classes[i]] = filtered_df
            return df_dict
        else:
            print("Error: Length of classes, keywords, and modes lists must be equal")
            sys.exit()
            
    def _make_dataset(self, dataset_dir, dataset_name, ds_dicts, id_col, start_col, end_col,
                      stream_index=0):
        dataset_path = os.path.join(dataset_dir, dataset_name)
        if not os.path.exists(dataset_path):
            os.mkdir(dataset_path)
        dset_count = 1
        for dsets in ds_dicts:
            print(f"\nMaking dataset {dset_count} of {len(ds_dicts)}...")
            class_dir = os.path.join(dataset_path, dsets)
            if os.path.exists(class_dir) == False:
                os.mkdir(class_dir)
            if len(ds_dicts[dsets]) == 0:
                print(f"\nWARNING: Dataset class {dsets} has no matching samples, class will be empty\n")
            ds_dicts[dsets].progress_apply(lambda row: youtube.download_waveform(row, class_dir,
                                                                             id_col=id_col,
                                                                             start_col=start_col,
                                                                             end_col=end_col,
                                                                             stream_index=stream_index,
                                                                             file_format=self.file_format),
                                                axis=1)
            n_queued_files = len(ds_dicts[dsets])
            n_downloaded_files = len(glob.glob(class_dir+"/*"))
            print(f"\n\t# downloads attempted = {n_queued_files}")
            print(f"\t# downloads failed = {n_queued_files - n_downloaded_files}")
            print(f"\t# downloads completed = {n_downloaded_files}")
            if n_downloaded_files != 0:
                print("Dataset made successfully!")
            else:
                print(f"WARNING: No files were successfully downloaded for dataset class {dsets}")
            dset_count += 1
        
    def _n_shards(self, n_samples):
        """Compute number of shards for number of samples.
        TFRecords are split into multiple shards. Each shard's size should be
        between 100 MB and 200 MB according to the TensorFlow documentation.
        Parameters
        ----------
        n_samples : int
            The number of samples to split into TFRecord shards.
        Returns
        -------
        n_shards : int
            The number of shards needed to fit the provided number of samples.
        """
        return math.ceil(n_samples / self._shard_size())
    
    def _shard_size(self):
        """Compute the shard size.
        Computes how many audio files with the given sample-rate and duration
        fit into one TFRecord shard to stay within the 100 MB - 200 MB limit.
        Returns
        -------
        shard_size : int
            The number samples one shard can contain.
        """
        shard_max_bytes = self.max_shard_size * 10**6
        shard_size = shard_max_bytes // self.filesize
        return int(shard_size * self.compression_scaling_factor)
    
    def _get_shard_path(self, shard_id):
        if not os.path.exists(self.shard_loc):
            os.mkdir(self.shard_loc)
        shard_path = os.path.join(self.shard_loc, f"{self.shard_name}_{shard_id}.tfrec")
        return shard_path
    
    def _get_filenames(self):
        self.ds_filenames = set(glob.glob(self.dataset_dir + "/*/*"))
        shard_files = set(glob.glob(self.shard_loc + "/*"))
        self.ds_filenames -= shard_files
        self.ds_filenames = list(self.ds_filenames)
        if len(self.ds_filenames) == 0:
            print("\nWARNING: No files in dataset, exiting...\n")
            sys.exit()
        random.shuffle(self.ds_filenames)
        self.n_files = len(self.ds_filenames)
        
    def _get_classes(self):
        self.classes = set(glob.glob(self.dataset_dir + "/*"))
        shard_dir = set(glob.glob(self.shard_loc))
        self.classes -= shard_dir
        self.classes = list(self.classes)
        self.classes = [(os.path.normpath(i)).split(os.sep)[-1] for i in self.classes]
    
    def _split_data_into_shards(self):
        files_per_shard = self._shard_size()
        shards = []
        shard_id = 0
        for size in range(0, self.n_files, files_per_shard):
            shard_path = self._get_shard_path(shard_id)
            shards.append((shard_path, self.ds_filenames[size : size + files_per_shard]))
            shard_id += 1
        return shards
    
    def _write_tfrecord_file(self, shard_data):
        shard_path, files = shard_data
        if os.path.exists(self.shard_loc):
            # if len(os.listdir(self.shard_loc)) != 0:
            #     shutil.rmtree(self.shard_loc)
            #     os.mkdir(self.shard_loc)
            pass
        else:
            os.mkdir(self.shard_loc)
        with tf.io.TFRecordWriter(shard_path, options='ZLIB') as out:
            for f in tqdm(files):
                features = {}
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    audio, sr = librosa.load(f, sr=self.sample_rate, mono=True,
                                             duration=self.duration)
                    if len(audio) < self.n_data_points:
                        audio = np.pad(audio, (0, self.n_data_points-len(audio)),
                                       mode="constant", constant_values=0)
                features['raw_audio'] = _float_feature(audio)
                label = tf.strings.split(f, os.sep)[-2]
                label_str = label.numpy().decode()
                self.label_int = self._encode_label(label_str)
                features['label'] = _int_feature(self.label_int)
                example = tf.train.Example(features=tf.train.Features(feature=features))
                out.write(example.SerializeToString())

    def convert(self):
        """Convert to TFRecords."""
        print("\nConverting audio files to TFRecord shards...")
        shards = self._split_data_into_shards()
        # _parallelize(self._write_tfrecord_file, shards)
        for s in shards:
            self._write_tfrecord_file(s)

        print("\nConversion to TFRecord shards complete!\n")
        print(f"Total number of audio files written to TFRecord: {self.n_files}")
        print(f"Number of TFRecord shards: {self._n_shards(self.n_files)}")
        print(f"TFRecord files saved to: {os.path.join(self.dataset_dir, self.shard_dir_name)}")
        
    def _parse_TFRecord(self, example, feature_description):     
        x = tf.io.parse_single_example(example, feature_description)
        return x['raw_audio'], x['label']
    
    def _encode_label(self, label):
        return self.label_encoder.transform([label])[0]
    
    def _decode_label(self, label_encoding):
        return self.label_encoder.inverse_transform([label_encoding])[0]

    def _prep_print_record(self, sample):
        audio, label = sample
        self.a = audio
        self.l = label
        print(audio.values.numpy(), label.numpy())

    def _raw_waveform(self, audio, label):
        return audio, label

    def _get_spectrogram(self, audio, n_fft=2048, hop_length=None, win_length=None,
                         window="hann", center=True, dtype=None, pad_mode="reflect"):
        return librosa.stft(audio, n_fft=n_fft, hop_length=hop_length, win_length=win_length,
                            window=window, center=center, dtype=dtype, pad_mode=pad_mode)

    def _get_mel_spectrogram(self, audio=None, sr=44100, S=None, n_fft=2048, hop_length=512,
                             win_length=None, window='hann', center=True,
                             pad_mode='reflect', power=2.0):
        return librosa.feature.melspectrogram(y=audio, sr=sr, S=S, n_fft=n_fft,
                                              hop_length=hop_length, win_length=win_length,
                                              window=window, center=center,
                                              pad_mode=pad_mode)

    def _prep_waveform(self, audio, label):
        audio_np = audio.numpy()
        aud = tf.convert_to_tensor(audio_np, dtype=tf.float32)
        aud = tf.reshape(aud, list(self.waveform_shape))
        return aud, label

    def _map_prep_waveform(self, audio, label):
        aud, lab = tf.py_function(self._prep_waveform, [audio.values, label], [tf.float32, tf.int64])
        aud.set_shape(tf.TensorShape(list(self.waveform_shape)))
        lab.set_shape(tf.TensorShape([]))
        return aud, lab

    def _waveform2spect(self, audio, label):
        audio_np = audio.numpy()
        spect = self._get_spectrogram(audio_np, n_fft=self.nfft,
                                           hop_length=self.hop_length,
                                           win_length=self.win_length,
                                           window=self.window,
                                           center=self.center,
                                           dtype=self.spect_dtype,
                                           pad_mode=self.pad_mode)
        spect = tf.convert_to_tensor(spect, dtype=tf.float32)
        spect = tf.reshape(spect, list(self.spect_shape))
        return spect, label

    def _map_waveform2spect(self, audio, label):
        spect, lab = tf.py_function(self._waveform2spect, [audio.values, label], [tf.float32, tf.int64])
        spect.set_shape(tf.TensorShape(list(self.spect_shape)))
        lab.set_shape(tf.TensorShape([]))
        return spect, lab
    
    def _waveform2melspect(self, audio, label):
        audio_np = audio.numpy()
        spect = self._get_spectrogram(audio_np, n_fft=self.nfft,
                                           hop_length=self.hop_length,
                                           win_length=self.win_length,
                                           window=self.window,
                                           center=self.center,
                                           dtype=self.spect_dtype,
                                           pad_mode=self.pad_mode)
        mel_spect = self._get_mel_spectrogram(sr=self.sample_rate, S=spect,
                                              n_fft=self.nfft,
                                              hop_length=self.hop_length,
                                              win_length=self.win_length,
                                              window=self.window,
                                              center=self.center,
                                              pad_mode=self.pad_mode,
                                              power=self.power)
        mel_spect = tf.convert_to_tensor(mel_spect, dtype=tf.float32)
        mel_spect = tf.reshape(mel_spect, list(self.spect_shape))
        return mel_spect, label

    def _map_waveform2melspect(self, audio, label):
        mel_spect, lab = tf.py_function(self._waveform2melspect, [audio.values, label], [tf.float32, tf.int64])
        mel_spect.set_shape(tf.TensorShape(list(self.spect_shape)))
        lab.set_shape(tf.TensorShape([]))
        return mel_spect, lab
                
    def read_shards(self):
        print("\nReading TFRecord shards...")
        AUTOTUNE = tf.data.experimental.AUTOTUNE
        shard_files = glob.glob(self.shard_loc + "/*")
        if len(shard_files) == 0:
            print("\nNo shard files exist for this dataset directory\nPlease run Dataset.convert() first")
            sys.exit()
        files_ds = tf.data.Dataset.list_files(shard_files)
        files_ds = files_ds.shuffle(buffer_size=len(shard_files), seed=self.random_seed,
                                    reshuffle_each_iteration=False)

        # Disregard data order in favor of reading speed
        ignore_order = tf.data.Options()
        ignore_order.experimental_deterministic = False
        files_ds = files_ds.with_options(ignore_order)
    
        # Read TFRecord files in an interleaved order
        ds = tf.data.TFRecordDataset(files_ds,
                                     compression_type='ZLIB',
                                     num_parallel_reads=AUTOTUNE)

        # get the number of files in the dataset
        self.ds_size = 0
        for count in ds:
            self.ds_size += 1
            
        # build the features dictionary for the shard in question
        self.feature_description = {}
        for record in ds.take(1):
            example = tf.train.Example()
            example.ParseFromString(record.numpy())
            
        for key, feature in example.features.feature.items():
        # The values are the Feature objects which contain a `kind` which contains:
        # one of three fields: bytes_list, float_list, int64_list
            kind = feature.WhichOneof('kind')
            if kind == "bytes_list":
                self.feature_description[key] = tf.io.FixedLenFeature([], tf.string)
            elif kind == "float_list":
                self.feature_description[key] = tf.io.VarLenFeature(tf.float32)
            elif kind == "int64_list":
                self.feature_description[key] = tf.io.FixedLenFeature([], tf.int64)
    
        # Prepare batches
        # ds = ds.batch(self.batch_size)

        # Parse a batch into a dataset of [audio, label] pairs
        ds = ds.map(lambda x: self._parse_TFRecord(x, self.feature_description))
        if self.final_format == "waveform":
            ds = ds.map(self._map_prep_waveform, num_parallel_calls=AUTOTUNE)
        elif self.final_format == "spect":
            ds = ds.map(self._map_waveform2spect, num_parallel_calls=AUTOTUNE)
        elif self.final_format == "mel_spect":
            ds = ds.map(self._map_waveform2melspect, num_parallel_calls=AUTOTUNE)

        print("TFRecord shards read successfully!")
        return ds

    def prep_data(self, dataset):
        print("\nPrepping data for ML use...\nLoading dataset dictionary...")

        print(f"Dataset size = {self.ds_size}")
        print(f"Train % = {self.train_cut * 100}")
        print(f"Validation % = {self.val_cut * 100}")
        print(f"Test % = {self.test_cut * 100}")
        print(f"# Epochs = {self.n_epochs}")
        print(f"# Labels = {len(self.classes)}")
        AUTOTUNE = tf.data.experimental.AUTOTUNE
        data_dict = {}

        # split into three parts
        assert (self.train_cut + self.val_cut + self.test_cut) == 1

        dataset = dataset.shuffle(buffer_size=self.buffer_size, seed=self.random_seed,
                                  reshuffle_each_iteration=False)

        train_size = int(self.train_cut * self.n_files)
        val_size = int(self.val_cut * self.n_files)
        
        train_ds = dataset.take(train_size)
        train_ds = train_ds.batch(self.batch_size)
        train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)
        data_dict['train_ds'] = train_ds

        val_ds = dataset.skip(train_size).take(val_size)
        val_ds = val_ds.batch(self.val_test_batch_size)
        val_ds = val_ds.prefetch(buffer_size=AUTOTUNE)
        data_dict['val_ds'] = val_ds

        test_ds = dataset.skip(train_size).skip(val_size)
        test_ds = test_ds.batch(self.val_test_batch_size)
        data_dict['test_ds'] = test_ds

        data_dict['n_epochs'] = self.n_epochs
        data_dict['n_labels'] = len(self.classes)
        data_dict['waveform_shape'] = self.waveform_shape
        if (self.final_format == "spect") or (self.final_format == "mel_spect"):
            data_dict['spect_shape'] = self.spect_shape

        data_dict['sample_rate'] = self.sample_rate

        print("Dataset dictionary prepared successfully!")

        return data_dict
        
if __name__ == '__main__':
    ds_dir = "C:\\Users\\ahste\\OneDrive\\ML Datasets\\AudioSet\\datasets\\"
    ds_name = "test_dataset2"
    params_file = "params.toml"
    import toml
    params = toml.load(params_file)
    ds_obj = Dataset(params)