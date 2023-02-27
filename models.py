# -*- coding: utf-8 -*-
"""
Created on Mon Dec 13 17:55:25 2021

@author: ahste
"""

import warnings
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    import os
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    from tensorflow.keras import models
    import datasets
    import sys
    from kapre import STFT, Magnitude, MagnitudeToDecibel, Delta, Frame
    from kapre.composed import get_melspectrogram_layer, get_log_frequency_spectrogram_layer, get_stft_magnitude_layer


class Model:
    def __init__(self, data_dict, dataset, params_dict):
        self.ds = dataset
        self.train_data = data_dict['train_ds']
        self.val_data = data_dict['val_ds']
        self.test_data = data_dict['test_ds']
        self.n_epochs = data_dict['n_epochs']
        self.n_labels = data_dict['n_labels']
        self.sample_rate = data_dict['sample_rate']
        self.waveform_shape = data_dict['waveform_shape']
        try:
            self.spect_shape = data_dict['spect_shape']
        except KeyError:
            pass

        # # get a sample of the training dataset and store input shape of first feature
        # for sample in ds.take(1):
        #     self.sample = sample
        #     try:
        #         self.input_shape = sample[0].numpy().shape
        #     except AttributeError:
        #         try:
        #             self.input_shape = sample[0].values.numpy().shape
        #         except Exception as e:
        #             print(e)
        #             sys.exit()

        self.params = params_dict
        self.learning_rate = self.params['model']['learning_rate']

    def m_waveform_classification(self):
        pass

    def spect_ae_basic(self, optimizer=tf.keras.optimizers.Adam(), loss=tf.keras.losses.MeanSquaredError,
                       metrics=['accuracy']):
        print(self.spect_shape)

        input_img = keras.Input(shape=self.spect_shape)

        x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(input_img)
        x = layers.MaxPooling2D((2, 2), padding='same')(x)
        x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
        x = layers.MaxPooling2D((2, 2), padding='same')(x)
        x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)
        encoded = layers.MaxPooling2D((2, 2), padding='same')(x)

        x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(encoded)
        x = layers.UpSampling2D((2, 2))(x)
        x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
        x = layers.UpSampling2D((2, 2))(x)
        x = layers.Conv2D(128, (3, 3), activation='relu')(x)
        x = layers.UpSampling2D((2, 2))(x)
        decoded = layers.Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)

        model = keras.Model(input_img, decoded)

        model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

        return model
    
    def m_spect_classification(self, optimizer=tf.keras.optimizers.Adam(),
                               loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                               metrics=['accuracy']):
        model = models.Sequential([
            layers.InputLayer(input_shape=self.spect_shape),
            # Downsample the input.
            layers.experimental.preprocessing.Resizing(32, 32),
            layers.Conv2D(32, 3, activation='relu', data_format="channels_last"),
            layers.Conv2D(64, 3, activation='relu'),
            layers.MaxPooling2D(),
            layers.Dropout(0.25),
            layers.Flatten(),
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(self.n_labels),
            ])

        model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

        return model

    def kapre_mel_spect_classification(self, optimizer=tf.keras.optimizers.Adam(),
                                       loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                                       metrics=["accuracy"]):
        print("\nCompiling model...")
        input_ = layers.Input(shape=self.waveform_shape)

        mel_layer = get_melspectrogram_layer(n_fft=self.params['model']['kapre_mel_spect']['n_fft'],
                                             win_length=self.params['model']['kapre_mel_spect']['win_length'],
                                             hop_length=self.params['model']['kapre_mel_spect']['hop_length'],
                                             window_name=self.params['model']['kapre_mel_spect']['window_name'],
                                             pad_begin=self.params['model']['kapre_mel_spect']['pad_begin'],
                                             pad_end=self.params['model']['kapre_mel_spect']['pad_end'],
                                             sample_rate=self.sample_rate,
                                             n_mels=self.params['model']['kapre_mel_spect']['n_mels'],
                                             mel_f_min=self.params['model']['kapre_mel_spect']['mel_f_min'],
                                             mel_f_max=self.params['model']['kapre_mel_spect']['mel_f_max'],
                                             mel_htk=self.params['model']['kapre_mel_spect']['mel_htk'],
                                             mel_norm=self.params['model']['kapre_mel_spect']['mel_norm'],
                                             return_decibel=self.params['model']['kapre_mel_spect']['return_decibel'],
                                             db_amin=self.params['model']['kapre_mel_spect']['db_amin'],
                                             db_ref_value=self.params['model']['kapre_mel_spect']['db_ref_value'],
                                             db_dynamic_range=self.params['model']['kapre_mel_spect']['db_dynamic_range'],
                                             input_data_format=self.params['model']['kapre_mel_spect']['input_data_format'],
                                             output_data_format=self.params['model']['kapre_mel_spect']['output_data_format'])(input_)

        delta_layer = Delta(win_length=self.params['model']['kapre_delta']['win_length'],
                            mode=self.params['model']['kapre_delta']['mode'],
                            data_format=self.params['model']['kapre_delta']['data_format'])(mel_layer)

        concat = layers.Concatenate()([mel_layer, delta_layer])

        frame_layer = Frame(frame_length=self.params['model']['kapre_frame']['frame_length'],
                            hop_length=self.params['model']['kapre_frame']['hop_length'],
                            pad_end=self.params['model']['kapre_frame']['pad_end'],
                            pad_value=self.params['model']['kapre_frame']['pad_value'],
                            data_format=self.params['model']['kapre_frame']['data_format'])(concat)

        x = layers.Conv2D(filters=self.params['model']['kapre_ms_classification']['cnn_block1']['filters'],
                          kernel_size=(self.params['model']['kapre_ms_classification']['cnn_block1']['kernel_size'][0],
                                       self.params['model']['kapre_ms_classification']['cnn_block1']['kernel_size'][1]),
                          padding=self.params['model']['kapre_ms_classification']['cnn_block1']['padding'])(frame_layer)
        x = layers.BatchNormalization()(x)
        x = layers.Activation(activation=self.params['model']['kapre_ms_classification']['cnn_block1']['activation'])(x)
        x = layers.Dropout(rate=self.params['model']['kapre_ms_classification']['cnn_block1']['dropout_rate'],
                           seed=self.params['data']['random_seed'])(x)
        x = layers.MaxPooling3D(pool_size=(self.params['model']['kapre_ms_classification']['cnn_block1']['pool_size'][0],
                                           self.params['model']['kapre_ms_classification']['cnn_block1']['pool_size'][1],
                                           self.params['model']['kapre_ms_classification']['cnn_block1']['pool_size'][2]))(x)

        x = layers.Conv2D(filters=self.params['model']['kapre_ms_classification']['cnn_block2']['filters'],
                          kernel_size=(self.params['model']['kapre_ms_classification']['cnn_block2']['kernel_size'][0],
                                       self.params['model']['kapre_ms_classification']['cnn_block2']['kernel_size'][1]),
                          padding=self.params['model']['kapre_ms_classification']['cnn_block2']['padding'])(frame_layer)
        x = layers.BatchNormalization()(x)
        x = layers.Activation(activation=self.params['model']['kapre_ms_classification']['cnn_block2']['activation'])(x)
        x = layers.Dropout(rate=self.params['model']['kapre_ms_classification']['cnn_block2']['dropout_rate'],
                           seed=self.params['data']['random_seed'])(x)
        x = layers.MaxPooling3D(pool_size=(self.params['model']['kapre_ms_classification']['cnn_block2']['pool_size'][0],
                                           self.params['model']['kapre_ms_classification']['cnn_block2']['pool_size'][1],
                                           self.params['model']['kapre_ms_classification']['cnn_block2']['pool_size'][2]))(x)

        x = layers.Conv2D(filters=self.params['model']['kapre_ms_classification']['cnn_block3']['filters'],
                          kernel_size=(self.params['model']['kapre_ms_classification']['cnn_block3']['kernel_size'][0],
                                       self.params['model']['kapre_ms_classification']['cnn_block3']['kernel_size'][1]),
                          padding=self.params['model']['kapre_ms_classification']['cnn_block3']['padding'])(frame_layer)
        x = layers.BatchNormalization()(x)
        x = layers.Activation(activation=self.params['model']['kapre_ms_classification']['cnn_block3']['activation'])(x)
        x = layers.Dropout(rate=self.params['model']['kapre_ms_classification']['cnn_block3']['dropout_rate'],
                           seed=self.params['data']['random_seed'])(x)
        x = layers.MaxPooling3D(pool_size=(self.params['model']['kapre_ms_classification']['cnn_block3']['pool_size'][0],
                                           self.params['model']['kapre_ms_classification']['cnn_block3']['pool_size'][1],
                                           self.params['model']['kapre_ms_classification']['cnn_block3']['pool_size'][2]))(x)

        x = layers.Conv2D(filters=self.params['model']['kapre_ms_classification']['cnn_block4']['filters'],
                          kernel_size=(self.params['model']['kapre_ms_classification']['cnn_block4']['kernel_size'][0],
                                       self.params['model']['kapre_ms_classification']['cnn_block4']['kernel_size'][1]),
                          padding=self.params['model']['kapre_ms_classification']['cnn_block4']['padding'])(frame_layer)
        x = layers.BatchNormalization()(x)
        x = layers.Activation(activation=self.params['model']['kapre_ms_classification']['cnn_block4']['activation'])(x)
        x = layers.Dropout(rate=self.params['model']['kapre_ms_classification']['cnn_block4']['dropout_rate'],
                           seed=self.params['data']['random_seed'])(x)
        x = layers.GlobalMaxPooling3D()(x)

        x = layers.Dropout(rate=self.params['model']['kapre_ms_classification']['final_dense']['dropout_rate'],
                           seed=self.params['data']['random_seed'])(x)
        x = layers.Dense(units=self.params['model']['kapre_ms_classification']['final_dense']['num_units'],
                         activation=self.params['model']['kapre_ms_classification']['final_dense']['activation'])(x)
        x = layers.Dense(units=self.n_labels,
                         activation=self.params['model']['kapre_ms_classification']['final_dense']['final_activation'])(x)

        model = keras.Model(inputs=[input_], outputs=[x], name="kapre_mel_spect_classification")

        model.compile(optimizer=tf.keras.optimizers.Adam(self.learning_rate), loss=loss, metrics=metrics)

        print("Model compiled successfully!")

        return model

    def kapre_mel_spect_ae_basic(self, optimizer=tf.keras.optimizers.Adam(),
                                       loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                                       metrics=["accuracy"]):
        print("\nCompiling model...")
        input_ = layers.Input(shape=self.waveform_shape)

        mel_layer = get_melspectrogram_layer(n_fft=self.params['model']['kapre_mel_spect']['n_fft'],
                                             win_length=self.params['model']['kapre_mel_spect']['win_length'],
                                             hop_length=self.params['model']['kapre_mel_spect']['hop_length'],
                                             window_name=self.params['model']['kapre_mel_spect']['window_name'],
                                             pad_begin=self.params['model']['kapre_mel_spect']['pad_begin'],
                                             pad_end=self.params['model']['kapre_mel_spect']['pad_end'],
                                             sample_rate=self.sample_rate,
                                             n_mels=self.params['model']['kapre_mel_spect']['n_mels'],
                                             mel_f_min=self.params['model']['kapre_mel_spect']['mel_f_min'],
                                             mel_f_max=self.params['model']['kapre_mel_spect']['mel_f_max'],
                                             mel_htk=self.params['model']['kapre_mel_spect']['mel_htk'],
                                             mel_norm=self.params['model']['kapre_mel_spect']['mel_norm'],
                                             return_decibel=self.params['model']['kapre_mel_spect']['return_decibel'],
                                             db_amin=self.params['model']['kapre_mel_spect']['db_amin'],
                                             db_ref_value=self.params['model']['kapre_mel_spect']['db_ref_value'],
                                             db_dynamic_range=self.params['model']['kapre_mel_spect'][
                                                 'db_dynamic_range'],
                                             input_data_format=self.params['model']['kapre_mel_spect'][
                                                 'input_data_format'],
                                             output_data_format=self.params['model']['kapre_mel_spect'][
                                                 'output_data_format'])(input_)



        model = keras.Model(inputs=[input_], outputs=[x], name="kapre_mel_spect_classification")

        model.compile(optimizer=tf.keras.optimizers.Adam(self.learning_rate), loss=loss, metrics=metrics)

        print("Model compiled successfully!")

        return model

    def fit_model(self, model, callbacks=None):
        print("\nFitting model...")
        return model.fit(self.train_data, epochs=self.n_epochs, validation_data=self.val_data,
                         callbacks=callbacks)


if __name__ == '__main__':
    import toml
    params_file = "params.toml"

    params = toml.load(params_file)

    ds_obj = datasets.Dataset(params)
    dd = ds_obj.dd
    ds = ds_obj.ds
    model_obj = Model(dd, ds, params)
    class_model = model_obj.spect_ae_basic()
    class_model_fit_metrics = model_obj.fit_model(class_model)