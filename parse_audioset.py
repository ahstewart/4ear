# -*- coding: utf-8 -*-
"""
Created on Thu Oct 28 16:40:30 2021

@author: ahste
"""

import pandas as pd
import json
from youtube import get_aud
import sys

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
    row_names = [str('"' + ont[i.strip()] + '"') for i in row_l if isinstance(i, str)]
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
    Returns the same DataFrame as load_data() with an extra "labels" column
    added onto the end.
    """
    data = load_data(csv_loc, csv_head)
    ont_dict = ont(ont_loc)
    if label_ix_end == 0:
        data[new_col_name] = labels(data.iloc[:,label_ix_start:], ont_dict)
    else:
        data[new_col_name] = labels(data.iloc[:,label_ix_start:label_ix_end], ont_dict)
    return data

def ont_subset2list(ont_subset_path):
    """
    Converts a subset of a JSON AudioSet Ontology into a list of labels.
    Returns the label list.
    For use in the filter_by_labels() function.
    """
    try:
        f = open(ont_subset_path)
        subset = json.load(f)
        subset_list = [i["name"].split(",") for i in subset[1:]]
        return [name.strip() for sublist in subset_list for name in sublist]
    except FileNotFoundError:
        print("JSON ontology subset file not found, check labels parameter. Should be a list of lists except for JSON paths")
        sys.exit()

def filter_by_labels(df_to_filter, labels=[], mode="or", label_col="labels"):
    """
    Returns a new DataFrame that is filtered by the given labels in label_col.
    Modes:
        or = selects rows that contain one or more of the provided labels
        and = selects rows that contain all of the provided labels
        only = selects rows that only contain the provided labels
        not = selects rows that do not contain any of the provided labels
    """
    if isinstance(labels, str):
        labels = ont_subset2list(labels)
    if mode == "or":
        dfs = list(range(len(labels)))
        count = 0
        for l in labels:
            dfs[count] = df_to_filter[df_to_filter[label_col].str.contains('"'+l+'"')]
            count += 1
        filtered_df = pd.concat(dfs)
    elif mode == "and":
        for l in labels:
            if labels.index(l) == 0:
                dfs = df_to_filter[df_to_filter[label_col].str.contains('"'+l+'"')]
            else:
                dfs = dfs[dfs[label_col].str.contains('"'+l+'"')]
        filtered_df = dfs
    elif mode == "only":
        # sort target labels alphabetically then join into a commma-separated list
        label_list = ['"'+i+'"' for i in labels]
        label_list.sort()
        label_str = ", ".join(label_list)
        # create new column of sorted string of labels
        df_to_filter["temp_label_list"] = [", ".join(sorted(i.split(", "))) for i in df_to_filter[label_col]]
        filtered_df = df_to_filter[df_to_filter["temp_label_list"] == label_str]
        filtered_df = filtered_df.drop(columns=["temp_label_list"])
    elif mode == "not":
        for l in labels:
            if labels.index(l) == 0:
                dfs = df_to_filter[~df_to_filter[label_col].str.contains('"'+l+'"')]
            else:
                dfs = dfs[~dfs[label_col].str.contains('"'+l+'"')]
            filtered_df = dfs
    return filtered_df
    
    
if __name__ == "__main__":
    parsed_df = parse_csv("C:\\Users\\ahste\\OneDrive\\ML Datasets\\AudioSet\\balanced_train_segments.csv", 
                          "C:\\Users\\ahste\\OneDrive\\ML Datasets\\AudioSet\\ontology-master\\ontology.json", 
                          csv_head=2, 
                          label_ix_start=3, 
                          label_ix_end=0,
                          new_col_name="labels")