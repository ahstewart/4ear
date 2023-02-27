# -*- coding: utf-8 -*-
"""
Created on Tue Nov  9 18:22:25 2021

@author: ahste
"""

import os
import params
from datasets import Dataset
from models import Model

params_file = "params.toml"

params = params.parse_params(params_file)

# load dataset
ds_obj = Dataset(params)
dd = ds_obj.dd
ds = ds_obj.ds

# input dataset objects into Model
##model_obj = Model(dd, ds, params)
##class_model = model_obj.kapre_mel_spect_classification()
##class_model_fit_metrics = model_obj.fit_model(class_model)