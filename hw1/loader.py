from os import path

import pandas as pd

import config


def load_training_dataset(file) -> pd.DataFrame:
    data_frame = pd.read_csv(path.join(config.training_dir, file), header=None)
    return data_frame


def load_normalized_dataset(file) -> pd.DataFrame:
    data_frame = pd.read_csv(path.join(config.normalized_dir, file), header=None)
    return data_frame


def load_test_dataset(file) -> pd.DataFrame:
    data_frame = pd.read_csv(path.join(config.normalized_dir, file), header=None)
    temp = data_frame.copy()
    temp.drop(temp.columns[len(temp.columns) - 1], axis=1, inplace=True)
    temp.insert(0, None, 1, True)
    return temp


def load_values(file) -> pd.DataFrame:
    data_frame = pd.read_csv(path.join(config.normalized_dir, file), header=None)
    res = data_frame[data_frame.columns[-1]].to_numpy()
    return res


def load_vector(file) -> pd.DataFrame:
    data_frame = pd.read_csv(path.join(config.vectors_dir, file), header=None)
    return data_frame


def save_normalized_dataset(data_frame: pd.DataFrame, file):
    data_frame.to_csv(path.join(config.normalized_dir, file), header=False, index=False)
