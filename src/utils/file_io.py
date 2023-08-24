import os
import numpy as np
import pandas as pd

# Reads the clip-wise HR data that was computed and stored in the csv files (per video)
def get_hr_data(file_name):
    hr_df = pd.read_csv(config.HR_DATA_PATH + f"{file_name}.csv")

    return hr_df.values


def get_hr_data_cwt(file_name, path):
    hr = np.loadtxt(path / f"{file_name}.csv", delimiter=',')

    return hr

def get_hr_data_filtered(file_name, path):
    file_name = file_name.split(' ')[0]
    # hr = np.loadtxt(path + f"{file_name}.csv", delimiter=',')
    hr = np.loadtxt(path / f"{file_name}.npy", delimiter=',')

    return hr


def get_hr_data_stmaps(file_name, path):
    file_name = file_name.split(' ')[0]
    hr = np.load(path / f"{file_name}.npy")[0]

    return hr


# Read the raw signal from the ground truth csv and resample.
# Not be needed during the model as we will compute the HRs first-hand and use them directly instead of raw signals
def read_target_data(target_data_path, video_file_name):
    signal_data_file_path = os.path.join(target_data_path, f"{video_file_name} PPG.csv")
    signal_df = pd.read_csv(signal_data_file_path)

    return signal_df["Signal"].values, signal_df["Time"].values