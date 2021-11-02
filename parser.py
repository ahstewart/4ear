# -*- coding: utf-8 -*-
"""
Created on Thu Oct 28 16:40:30 2021

@author: ahste
"""

import numpy as np
import librosa
import pandas as pd
from pytube import YouTube
import json

def load_data(csv_path, csv_header):
    """
    Reads the audioset csv into a Pandas DataFrame.
    csv_header = the row that has the column headers (zero_indexed), default is 2
    """
    return pd.read_csv(csv_path, header=csv_header)
        
def ont(ont_path):
    """
    Loads the AudioSet Ontology into a list, then creates a dictionary with
    only the {id: name} key/value pair for each id.
    This is used to replace the coded IDs with the human-readable names.
    """
    ont_file = open("C:\\Users\\ahste\\OneDrive\\ML Datasets\\AudioSet\\ontology-master\\ontology.json")
    ont_json = json.load(ont_file)
    ont_ids = [i["id"] for i in ont_json]
    ont_names = [j["name"] for j in ont_json]
    ont_zip = zip(ont_ids, ont_names)
    return {k:v for (k,v) in ont_zip}

def labels2str(row, ont):
    """
    Creates a list from a row's labels (cols label1 -> labelX), then uses the
    list to return a comma-separated string of all the label names.
    NaN values are ignored.
    """
    row_l = list(row)
    row_names = [ont[i.strip()] for i in row_l if isinstance(i, str)]
    return ", ".join(row_names)

def labels(df, ont):
    """
    Calls labels2str() for each row in the dataset to create a new Series of
    comma-separated label name strings.
    A sliced version of the DataFrame is fed into this function, all rows but
    with the non-label columns removed.
    """
    return df.apply(lambda row: labels2str(row, ont), axis=1)
    
def parse_csv(csv_loc, ont_loc, csv_head=2, label_ix_start=3, label_ix_end=0, new_col_name="labels"):
    """
    Combines all of the above funcs to load the csv and ontology, create a new
    Series of label name strings, then tack on this new Series to the end of 
    the DataFrame.
    Returns the same DataFrame as load_data() with an extra "Labels" column
    added onto the end.
    """
    data = load_data(csv_loc, csv_head)
    ont_dict = ont(ont_loc)
    if label_ix_end == 0:
        data[new_col_name] = labels(data.iloc[:,label_ix_start:], ont_dict)
    else:
        data[new_col_name] = labels(data.iloc[:,label_ix_start:label_ix_end], ont_dict)
    return data
    

if __name__ == "__main__":
    parsed_df = parse_csv("C:\\Users\\ahste\\OneDrive\\ML Datasets\\AudioSet\\balanced_train_segments.csv", 
                          "C:\\Users\\ahste\\OneDrive\\ML Datasets\\AudioSet\\ontology-master\\ontology.json", 
                          csv_head=2, 
                          label_ix_start=3, 
                          label_ix_end=0,
                          new_col_name="labels")
