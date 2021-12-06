# -*- coding: utf-8 -*-
"""
Created on Tue Nov  9 18:22:25 2021

@author: ahste
"""

import toml
import os
from datasets import dataset_dicts, make_dataset
from parse_audioset import parse_csv

params_file = "params.toml"

params = toml.load(params_file)

# get parsing parameters
id_column = params["df_parsing"]["id_column"]
start_column = params["df_parsing"]["start_column"]
end_column = params["df_parsing"]["end_column"]

# get dataset parameters
create_dataset = params["dataset"]["create_dataset"]
dataset_name = params["dataset"]["dataset_name"]
dataset_dir = params["dataset"]["dataset_dir"]
dataset_format = params["dataset"]["format"]
dataset_classes = params["dataset"]["classes"]
use_ont_subset = params["dataset"]["use_ont_subset"]
pull_keywords = params["dataset"]["pull_keywords"]
pull_modes = params["dataset"]["pull_modes"]
label_column = params["dataset"]["label_column"]

# define data directory
data_dir_loc = params["paths"]["data_dir_loc"]
if os.path.exists(data_dir_loc) == False:
    os.mkdir(data_dir_loc)

# get data
parsed_df = parse_csv("C:\\Users\\ahste\\OneDrive\\ML Datasets\\AudioSet\\balanced_train_segments.csv", 
                      "C:\\Users\\ahste\\OneDrive\\ML Datasets\\AudioSet\\ontology-master\\ontology.json", 
                      csv_head=2, 
                      label_ix_start=3, 
                      label_ix_end=0,
                      new_col_name="labels")

# create dataset
if create_dataset == True:
    print(f"\nCreating new dataset in {dataset_dir}\n")
    data_dicts = dataset_dicts(parsed_df, classes=dataset_classes, 
                               keywords=pull_keywords, modes=pull_modes, 
                               label_col=label_column)
    make_dataset(dataset_dir, dataset_name, data_dicts, id_col=id_column, 
                 start_col=start_column, end_col=end_column, stream_index=0)
    
