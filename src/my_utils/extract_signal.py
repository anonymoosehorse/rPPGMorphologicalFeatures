import json
import numpy as np
from scipy.signal import butter,filtfilt
import matplotlib.pyplot as plt
import glob
from tqdm import tqdm

import config


def pos(r, g, b):
    tn_r = temp_normalize(r)
    tn_g = temp_normalize(g)
    tn_b = temp_normalize(b)

    s1 = tn_g - tn_b
    s2 = tn_g + tn_b - 2 * tn_r

    alpha = np.nanstd(s1) / np.nanstd(s2)

    z = s1 + alpha * s2

    return z


def temp_normalize(signal):
    #print(signal[0])
    #print(np.nan(signal))
    mean = np.nanmean(signal)
    #print(mean)
    return signal / (mean - 1)


def butter_lowpass_filter(data, fs, order):
    nyq = 0.5 * fs
    # Get the filter coefficients
    # b, a = butter(order, normal_cutoff, btype='low', analog=False)
    b, a = butter(order, (0.5 / nyq, 6/ nyq), btype='bandpass', analog=False)
    y = filtfilt(b, a, data)

    return y


def plot_signal(traces_path):
    files = glob.glob(traces_path + "*.json")

    for file in tqdm(files, total=len(files)):
        file = file.replace('\\', '/')

        with open(file, 'r') as f:
            data = json.load(f)

        r = np.array(data["R"], dtype=float)
        g = np.array(data["G"], dtype=float)
        b = np.array(data["B"], dtype=float)
        t = np.array(data["Times"], dtype=float)

        r_norm = (r - r.min()) / (r.max() - r.min())
        g_norm = (g - g.min()) / (g.max() - g.min())
        b_norm = (b - b.min()) / (b.max() - b.min())

        plt.plot(t, r_norm, 'r')
        plt.plot(t, g_norm, 'g')
        plt.plot(t, b_norm, 'b')
        plt.show()


def extract_signal_stmap(data_path, save_path, fps):
    files = glob.glob(data_path + "*_*.npy")

    nan_count = 0
    print(f"Extracting signal for {len(files)} files...")
    for file in tqdm(files):
        file = file.replace('\\', '/')
        data = np.load(file)
        data = np.swapaxes(data, 0, 1)
        data_new = np.zeros((data.shape[0], data.shape[1]))

        for i in range(len(data)):
            data_new[i] = pos(data[i, :, 0], data[i, :, 1], data[i, :, 2])
            data_new[i] = butter_lowpass_filter(data_new[i], fps, 2)

        nan = np.sum(np.isnan(data_new))
        if nan > 0:
            nan_count += 1

        output = np.nan_to_num(data_new)

        #plt.plot(np.arange(0, 250, 1), output[100])
        #plt.show()

        fname = file.split('/')[-1]

        np.savetxt(save_path + fname, output, delimiter=',')

    print(f"Found NaN in {nan_count} of {len(files)} files")


def extract_signal(traces_path, save_path, fps, dataset):
    files = glob.glob(traces_path + "*.json")
    nan_count = 0
    print(f"Extracting signal for {len(files)} files...")
    for file in tqdm(files, total=len(files)):
        file = file.replace('\\', '/')
        if dataset == "vipl" and file.split("/")[-1].split(" ")[0].endswith("source1"):
            fps = 25

        with open(file, 'r') as f:
            data = json.load(f)

        r = np.array(data["R"], dtype=float)
        g = np.array(data["G"], dtype=float)
        b = np.array(data["B"], dtype=float)
        t = np.array(data["Times"], dtype=float)

        signal = pos(r, g, b)

        #nan = np.argwhere(np.isnan(signal))[:,0]
        #output = np.stack((t, signal))
        #output = np.delete(output, nan, axis=1)

        nan = np.argwhere(np.isnan(signal))[:, 0]
        if len(nan) > 0:
            nan_count += 1

        np.nan_to_num(signal, copy=False)

        output = np.stack((t, signal))

        # plt.plot(output[0,:300], output[1, :300])

        if len(output[1]) < 10:
            print(file)
            continue
        output[1] = butter_lowpass_filter(output[1], fps, 2)

        #plt.plot(output[0, :300], output[1, :300])
        #plt.show()

        fname = file.split('/')[-1].split('.')[0]

        np.savetxt(save_path + fname + ".csv", output, delimiter=',')

    print(f"Found NaN in {nan_count} of {len(files)} files")

if __name__ == '__main__':
    traces_path = config.VICAR_TRACES_PATH
    save_path = config.TRACES_FILTERED_PATH
    dataset = 'vipl'
    # traces_path = "C:/Users/ruben/Documents/thesis/data/vipl/raw_signal/"
    fps = 30

    # extract_signal(traces_path, save_path, fps, dataset)

    # data_path = "C:/Users/ruben/Documents/thesis/data/vipl/split_stmaps2/"
    data_path = config.SPLIT_STMAPS
    save_path = config.SPLIT_STMAPS_FILTERED

    extract_signal_stmap(data_path, save_path, fps)

    #plot_signal(traces_path)


