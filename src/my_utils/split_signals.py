import numpy as np
import torch

import config
import glob
from tqdm import tqdm
import matplotlib.pyplot as plt
import os.path
import torch.nn.functional as F
from torchvision.transforms import ToTensor, ToPILImage
import math
import os
import h5py

def detect_peaks(sig, peak_delta):
    peaks = []
    valleys = []

    # Normalization by the mean
    norm_sig = sig - np.mean(sig)
    delta = peak_delta * np.max(norm_sig)

    mxpos = 0
    mnpos = 0
    lookformax = True
    mx = np.NINF
    mn = np.Inf

    for (i, temp) in enumerate(norm_sig):
        if (temp > mx):
            mx = temp
            mxpos = i
        if (temp < mn):
            mn = temp
            mnpos = i

        if (lookformax):

            if (temp < (mx - delta)):
                # numPeaks++
                peaks.append(mxpos)
                mn = temp
                mnpos = i
                lookformax = False

        else:

            if (temp > (mn + delta)):
                # numValleys++
                valleys.append(mnpos)
                mx = temp
                mxpos = i
                lookformax = True

    # Remove the peak in clipped areas
    # for (int i = 0 i < (parameters.LeftClip - 1) i++) peakers[i] = false
    # for (int i = peakers.Length - (parameters.RightClip + 1) i < len(peakers) i++) peakers[i] = false

    return peaks, valleys


def split_signal(signal_path, signal_length, signal_save_path, hr_save_path):
    files = glob.glob(signal_path + "*.csv")
    print("Splitting signals for " + str(len(files)) + " files")

    for file in tqdm(files, total=len(files)):
        # data = np.loadtxt(file, delimiter=',', skiprows=1)
        data = np.loadtxt(file, delimiter=',')
        peaks, valleys = detect_peaks(data[1], 0.1)

        peak_times = [data[0, p] for p in peaks]
        valley_times = [data[0, v] for v in valleys]
        peak_heights = [data[1, p] for p in peaks]
        valley_heights = [data[1, v] for v in valleys]
        plt.plot(data[0], data[1])
        plt.plot(peak_times, peak_heights, 'r*')
        plt.plot(valley_times, valley_heights, 'g*')
        plt.show()

        if abs(len(peaks) - len(valleys)) > 1 or len(peaks) < 10 or len(valleys) < 10:
            print("-----------------")
            print("Amount of peaks: " + str(len(peaks)))
            print("Amount of valleys: " + str(len(valleys)))

        valley_split = [valleys[i] for i in range(len(valleys)) if i % signal_length == 0]
        split_sigs = np.split(data, valley_split)[1:]

        if len(valleys) - 1 % signal_length != 0:
            split_sigs = split_sigs[:-1]

        fname = file.split("/")[-1].split(".")[0]
        #for i in range(len(split_sigs)):
        #    np.savetxt(config.SPLIT_SIGNAL_DIR + fname + "_" + str(i) + ".csv", split_sigs[i], delimiter=",")


def gt_to_signal_peaks(signal, valley_times, peak_times):
    sig_valley_idx = []
    sig_peak_idx = []

    k = 0
    i = 0
    for j, x in enumerate(signal):
        if i < len(valley_times):
            if x > valley_times[i]:
                sig_valley_idx.append(j)
                i += 1
        if k < len(peak_times):
            if x > peak_times[k]:
                sig_peak_idx.append(j)
                k += 1

    return sig_valley_idx, sig_peak_idx


def calc_bpms_nopeaks(split_indices, signal_data, signal_length, fname):
    bpms = []
    for i in range(len(split_indices) - 1):
        bpms.append(60 / ((signal_data[0, split_indices[i + 1]] - signal_data[0, split_indices[i]]) / 1000) * signal_length)
    avg_bpm = sum(bpms) / len(bpms)
    if avg_bpm < 30 or avg_bpm > 120:
        print(fname)
        print(bpms)
    return bpms


def calc_bpms_peaks(split_times, sig_peak_times, fname, split_sigs):
    bpms = []
    peak_idx = 0
    i = 0
    # Iteratively go through every signal part and calculate the average bpm within that window
    while i < len(split_times) - 1:
        split_time = split_times[i+1]
        n_peaks = 0
        if peak_idx < len(sig_peak_times):
            t_first = sig_peak_times[peak_idx]

        # Count the amount of peaks within the time window
        while peak_idx <= len(sig_peak_times) - 1 and sig_peak_times[peak_idx] < split_time:
            n_peaks += 1
            peak_idx += 1
        if peak_idx == len(sig_peak_times) - 1:
            t_last = sig_peak_times[-1]
        else:
            t_last = sig_peak_times[peak_idx - 1]

        # If less than two peaks are found, the peak-to-peak time cannot be calculated
        if n_peaks <= 1:
            print(fname)
            print(f"Detected only {n_peaks} peak(s) between {split_times[i]} and "
                  f"{split_times[i+1]}")
            del split_sigs[i]
            del split_times[i+1]
        # Otherwise, calculate the bpm using the time between the first and last peak
        else:
            # bpms.append(60 / ((sig_peak_times[peak_idx - 1] - sig_peak_times[peak_idx - (n_peaks - 1)]) / 1000) * (n_peaks - 1))
            bpms.append(60 / ((t_last - t_first) / 1000) * (n_peaks - 1))

            i += 1
    avg_bpm = sum(bpms) / len(bpms)
    if avg_bpm < 30 or avg_bpm > 120:
        print(fname)
        print(bpms)

    return bpms


def test_bpmcalc():
    split_indices = list(np.arange(1, 1250, 250))
    sig_peak_times = np.concatenate((np.arange(100, 10000, 1000), np.arange(19100, 50000, 1000)))
    signal_data = np.array([np.arange(0, 50000, 40), np.arange(0, 50000, 40)])
    fname = "test"
    split_sigs = np.split(signal_data, split_indices, axis=1)[1:-1]

    bpms = calc_bpms_peaks(split_indices, sig_peak_times, signal_data, fname, split_sigs)

    split_times = [signal_data[0, idx] for idx in split_indices]
    #print(split_times)
    #print(bpms)
    #print(sig_peak_times)
    print(split_sigs)

    assert len(bpms) == 4
    for bpm in bpms:
        assert bpm == 60


def calc_bpm_windowsize(peak_times, window_time, split_sigs):
    n_windows = len(split_sigs)
    peaks_per_window = []
    bpms = []

    for i in range(n_windows):
        peaks_per_window.append([])

    for peak in peak_times:
        if peak/window_time < len(peaks_per_window):
            peaks_per_window[int(peak/window_time)].append(peak)

    deleted = 0
    for i in range(n_windows):
        if len(peaks_per_window[i]) > 1:
            peaks_per_second = (len(peaks_per_window[i]) - 1) / ((peaks_per_window[i][-1] - peaks_per_window[i][0]) / 1000)
            bpms.append(60 * peaks_per_second)
        else:
            del split_sigs[i - deleted]
            deleted += 1

    return bpms


def calc_bpm_stride(peak_times, window_time, split_sigs, stride):
    n_windows = len(split_sigs)
    peaks_per_window = []
    bpms = []

    for i in range(n_windows):
        peaks_per_window.append([])

    for peak in peak_times:
        first = max(0, int((peak - window_time) / stride))
        last = min(n_windows - 1, int(peak / stride))
        while first <= last:
            peaks_per_window[first].append(peak)
            first += 1

    deleted = 0
    for i in range(n_windows):
        if len(peaks_per_window[i]) > 1:
            peaks_per_second = (len(peaks_per_window[i]) - 1) / ((peaks_per_window[i][-1] - peaks_per_window[i][0]) / 1000)
            bpms.append(60 * peaks_per_second)
        else:
            del split_sigs[i - deleted]
            deleted += 1


    return bpms


def calc_pwa(valley_times, peak_times, valley_heights, peak_heights, signal_length, fname):
    amp_per_split = [[]]
    split_idx = 0
    peak_idx = 0
    for valley_idx in range(len(valley_times)):
        # Go to new split
        if valley_idx % signal_length == 0 and valley_idx != 0:
            avg_amp = sum(amp_per_split[split_idx]) / len(amp_per_split[split_idx])

            if len(amp_per_split[split_idx]) < 3:
                print(f"Invalid amount of peaks detected for {fname}: {len(amp_per_split[split_idx])}")

            amp_per_split[split_idx] = avg_amp
            amp_per_split.append([])
            split_idx += 1

        # Find the first peak after the current valley
        while peak_idx < len(valley_times) - 1 and peak_times[peak_idx] < valley_times[valley_idx]:
            peak_idx += 1
        # If this peak pairs with the valley, add the rise time
        if valley_idx < len(valley_times) - 1 and peak_times[peak_idx] < valley_times[valley_idx + 1]:
            amp_per_split[split_idx].append(peak_heights[peak_idx] - valley_heights[valley_idx])
            valley_idx += 1
            peak_idx += 1

    return amp_per_split


def calc_features(valley_times, peak_times, valley_heights, peak_heights, split_times, gt_data, valley_idxs, split_sigs, dataset):
    bpm_per_split = [[]]
    rt_per_split = [[]]
    pwa_per_split = [[]]
    area_per_split = [[]]
    split_idx = 0
    peak_idx = 0
    stop_loop = False
    for valley_idx, valley_time in enumerate(valley_times):
        # Go to new split
        while valley_time >= split_times[split_idx]:
            split_idx += 1
            if split_idx >= len(split_times):
                stop_loop = True
                break
            bpm_per_split.append([])
            rt_per_split.append([])
            pwa_per_split.append([])
            area_per_split.append([])

        #if valley_idx % signal_length == 0 and valley_idx != 0:
        #    rise_time = sum(rts_per_split[split_idx]) / len(rts_per_split[split_idx])
        #    if len(rts_per_split[split_idx]) < 3:
        #        print(f"Invalid amount of peaks detected for {fname}: {len(rts_per_split[split_idx])}")
        #    elif rise_time > 1000 or rise_time <= 50:
        #        print(f"Invalid rise time for {fname}: {rise_time} ms")
        #    rts_per_split[split_idx] = rise_time
        #    rts_per_split.append([])
        #    split_idx += 1

        # Find the first peak after the current valley
        while peak_times[peak_idx] < valley_times[valley_idx]:
            peak_idx += 1
            if peak_idx == len(peak_times):
                stop_loop = True
                break

        if stop_loop:
            break

        # If this peak pairs with the valley, add the rise time
        if valley_idx < len(valley_times) - 1 and peak_times[peak_idx] < valley_times[valley_idx + 1]:
            rt_per_split[split_idx].append(peak_times[peak_idx] - valley_times[valley_idx])
            pwa = peak_heights[peak_idx] - valley_heights[valley_idx]
            area = sum(gt_data[valley_idxs[valley_idx]:valley_idxs[valley_idx + 1]])
            area_norm = area / (valley_times[valley_idx + 1] - valley_times[valley_idx]) * 1000
            if dataset == 'vicar':
                area_norm = (area_norm / 30 - 10818) / 54595
                pwa = pwa / 54595
            else:
                area_norm = area_norm / 30

            pwa_per_split[split_idx].append(pwa)
            area_per_split[split_idx].append(area_norm)
            t = (valley_times[valley_idx+1] - valley_times[valley_idx]) / 1000
            bpm_per_split[split_idx].append(60 / t)
    i = 0
    while i < len(bpm_per_split):
        if len(bpm_per_split[i]) == 0:
            del bpm_per_split[i], rt_per_split[i], pwa_per_split[i], area_per_split[i], split_sigs[i]
            print(f"Not enough peaks detected: {i}")
        elif sum(bpm_per_split[i]) / len(bpm_per_split[i]) < 25 or sum(bpm_per_split[i]) / len(bpm_per_split[i]) > 240:
            print(f"Invalid HR: {sum(bpm_per_split[i]) / len(bpm_per_split[i])}")
            del bpm_per_split[i], rt_per_split[i], pwa_per_split[i], area_per_split[i], split_sigs[i]
        elif sum(rt_per_split[i]) / len(rt_per_split[i]) < 50 or sum(rt_per_split[i]) / len(rt_per_split[i]) > 1000:
            print(f"Invalid RT: {sum(rt_per_split[i]) / len(rt_per_split[i])}")
            del bpm_per_split[i], rt_per_split[i], pwa_per_split[i], area_per_split[i], split_sigs[i]
        else:
            bpm_per_split[i] = sum(bpm_per_split[i]) / len(bpm_per_split[i])
            rt_per_split[i] = sum(rt_per_split[i]) / len(rt_per_split[i])
            pwa_per_split[i] = sum(pwa_per_split[i]) / len(pwa_per_split[i])
            area_per_split[i] = sum(area_per_split[i]) / len(area_per_split)
            i += 1
    return bpm_per_split, rt_per_split, pwa_per_split, area_per_split


def split_stmap(signal_path, gt_path, signal_save_path, dataset, use_stride):
    files = glob.glob(signal_path + "*.npy")
    print("Splitting signals for " + str(len(files)) + " files")
    dt = []
    if not os.path.isdir(signal_save_path):
        print("Making directories")
        os.mkdir(signal_save_path)
        os.mkdir(signal_save_path[:-1] + "_hr")
        os.mkdir(signal_save_path[:-1] + "_rt")
        os.mkdir(signal_save_path[:-1] + "_pwa")
        os.mkdir(signal_save_path[:-1] + "_area")

    for file in tqdm(files, total=len(files)):
        file = file.replace('\\', '/')
        fname = file.split('/')[-1].split('.')[0]

        gt_name = fname.replace("s", "source")
        if os.path.isfile(gt_path + gt_name + ".h5") or os.path.isfile(gt_path + gt_name + ".csv") or os.path.isfile(gt_path + gt_name + " PPG.csv"):
            # Load ground truth signal
            if dataset == 'vicar':
                gt_t = 2
                fps = 30
                clip_rois = 264
                #gt_data = np.loadtxt(gt_path + gt_name + ".txt", skiprows=3)[:, 1:3]
                #gt_data = gt_data[np.where(gt_data[:, 0] == 1), 1][0]
                with h5py.File(gt_path + gt_name + ".h5", "r") as f:
                    gt_data = np.array(f['data']['PPG'])
            elif dataset == 'vipl':
                gt_t = 1000 / 60
                fps = 30
                clip_rois = 240
                gt_data = np.loadtxt(gt_path + gt_name + ".csv", delimiter=',', skiprows=1)[:, 1]
            else:
                gt_t = 16
                fps = 30
                clip_rois = 266
                gt_data = np.loadtxt(gt_path + gt_name + " PPG.csv", delimiter=',', skiprows=1)[:, 1]
            st_map = np.load(file)

            clip_top = math.floor((st_map.shape[1] - clip_rois) / 2)
            clip_bottom = math.ceil((st_map.shape[1] - clip_rois) / 2)
            if clip_bottom != 0:
                st_map = st_map[:, clip_top:-clip_bottom, :]

            if dataset == 'vipl' and fname.split('_')[-1] == "s1":
                st_map = ToTensor()(st_map).permute(0, 2, 1)
                st_map = F.interpolate(st_map, size=int(st_map.shape[2] * 30 / 25), mode='linear')
                st_map = st_map.permute(2, 1, 0).numpy()

            signal_t = 1000 / fps
            window_time = 10000
            signal_length = int(window_time / signal_t)
            gt_time = [i * gt_t for i in range(len(gt_data))]
            dt.append(len(st_map) * signal_t - len(gt_data) * gt_t)

            peaks, valleys = detect_peaks(gt_data, 0.3)

            all_valley_times = [gt_time[i] for i in valleys]
            all_peak_times = [gt_time[i] for i in peaks]
            all_valley_heights = [gt_data[i] for i in valleys]
            all_peak_heights = [gt_data[i] for i in peaks]

            #plt.plot(gt_time[0:int(len(gt_time)/4)], gt_data[0:int(len(gt_time)/4)])
            #plt.plot(all_valley_times[0:int(len(all_valley_times)/4)], all_valley_heights[0:int(len(all_valley_times)/4)], "go")
            #plt.plot(all_peak_times[0:int(len(all_peak_times)/4)], all_peak_heights[0:int(len(all_peak_times)/4)], "ro")
            #plt.show()

            split_times =[]
            if use_stride:
                offset = 0
                stride = int(0.5 * fps)
                split_sigs = []
                while offset + signal_length < len(st_map):
                    split_times.append(offset * signal_t)
                    split_sigs.append(st_map[offset:offset+signal_length])
                    offset += stride

                #bpms = calc_bpm_stride(all_peak_times, window_time, split_sigs, 500)
            else:
                split_indices = list(np.arange(0, len(st_map), signal_length))  # 10 seconds
                for idx in split_indices:
                    split_times.append(idx * signal_t)
                # split_times = split_times[:-1]

                split_sigs = np.split(st_map, split_indices, axis=0)[1:-1]
                #bpms = calc_bpm_windowsize(all_peak_times, window_time, split_sigs)

            if len(split_sigs) < 1:
                print(f"Signal duration is too small: {fname}")
                print(f"{st_map.shape}")
                continue
            split_times = split_times[1:]

            bpm_per_split, rt_per_split, pwa_per_split, area_per_split = calc_features(all_valley_times, all_peak_times,
                                        all_valley_heights, all_peak_heights, split_times, gt_data, valleys, split_sigs,
                                                                                       dataset)

            for i in range(len(bpm_per_split)):
                np.save(f"{signal_save_path}{fname}_{str(i)}.npy", split_sigs[i])
                np.save(f"{signal_save_path[:-1]}_hr/{fname}_{str(i)}.npy", [bpm_per_split[i]])
                np.save(f"{signal_save_path[:-1]}_rt/{fname}_{str(i)}.npy", [rt_per_split[i]])
                np.save(f"{signal_save_path[:-1]}_pwa/{fname}_{str(i)}.npy", [pwa_per_split[i]])
                np.save(f"{signal_save_path[:-1]}_area/{fname}_{str(i)}.npy", [area_per_split[i]])
        else:
            if fname.split('_')[-1] != "s2":
                print(f"GT-file not found: {gt_name}")

    #dt = np.asarray(dt)
    #q25, q75 = np.percentile(dt, [25, 75])
    #bin_width = 2 * (q75 - q25) * len(dt) ** (-1 / 3)
    #bins = round((dt.max() - dt.min()) / bin_width)
    #plt.hist(dt, bins=bins, density=True)
    #plt.savefig("dts_vipl2.png")





def split_signal2(signal_path, gt_path, signal_save_path, dataset, use_gt, use_stride):
    files = glob.glob(signal_path + "*.csv")
    print("Splitting signals for " + str(len(files)) + " files")
    highest = np.NINF
    lowest = np.Inf
    dt = []

    if not os.path.isdir(signal_save_path):
        print("Making directories")
        os.mkdir(signal_save_path)
        os.mkdir(signal_save_path[:-1] + "_hr")
        os.mkdir(signal_save_path[:-1] + "_rt")
        os.mkdir(signal_save_path[:-1] + "_pwa")
        os.mkdir(signal_save_path[:-1] + "_area")

    for file in tqdm(files, total=len(files)):
        file = file.replace('\\', '/')
        fname = file.split('/')[-1].split(' ')[0]

        if os.path.isfile(gt_path + fname + ".h5") or os.path.isfile(gt_path + fname + ".csv"):
            # Load ground truth signal
            if dataset == 'vicar':
                gt_fps = 500
                fps = 30
                #gt_data = np.loadtxt(gt_path + fname + ".txt", skiprows=3)[:, 1:3]
                #gt_data = gt_data[np.where(gt_data[:, 0] == 1), 1][0]
                with h5py.File(gt_path + fname + ".h5", "r") as f:
                    gt_data = np.array(f['data']['PPG'])
            else:
                gt_fps = 60
                fps = 30
                gt_data = np.loadtxt(gt_path + fname + ".csv", delimiter=',', skiprows=1)[:, 1]
            gt_t = 1000 / gt_fps
            gt_time = [i * gt_t for i in range(len(gt_data))]

            # Load pixel trace signal
            if use_gt:
                #signal_t = gt_t
                signal_t = 1000 / fps
                signal_data = np.asarray([gt_time, gt_data])
                signal_length = int(10 * 1000 / signal_t)

                signal_data = ToTensor()(signal_data)
                signal_data = F.interpolate(signal_data, size=int(len(signal_data[0, 0]) * 30 / gt_fps), mode='linear')
                signal_data = signal_data.numpy()[0]
            else:
                signal_t = 1000 / fps
                signal_data = np.loadtxt(file, delimiter=',')
                signal_length = int(10 * 1000 / signal_t)

                if dataset == 'vipl' and fname.split(' ')[0].endswith("source1"):
                    signal_data = ToTensor()(signal_data)
                    signal_data = F.interpolate(signal_data, size=int(len(signal_data[0, 0]) * 30 / 25), mode='linear')
                    signal_data = signal_data.numpy()[0]

            dt.append((signal_data[0, -1] - signal_data[0, 0]) - gt_time[-1])

            # Detect peaks and valleys
            peaks, valleys = detect_peaks(gt_data, 0.3)
            # peaks, valleys = detect_peaks(signal_data[1], 0.05)
            # valley_split = [valleys[i] for i in range(len(valleys)) if i % signal_length == 0]

            #peak_data = np.loadtxt(gt_path + fname + ".csv", delimiter=',', skiprows=1)[:, 2]  # FOR VIPL
            #peaks = np.argwhere(peak_data)
            #peaks = np.reshape(peaks, (len(peaks)))

            all_valley_times = [gt_time[i] for i in valleys]
            all_valley_heights = [gt_data[i] for i in valleys]
            all_peak_times = [gt_time[i] for i in peaks]
            all_peak_heights = [gt_data[i] for i in peaks]

            split_times = []
            if use_stride:
                offset = 0
                stride = int(0.5 * fps)
                split_sigs = []
                while offset + signal_length < len(signal_data[0]):
                    split_times.append(offset * signal_t)
                    split_sigs.append(signal_data[:, offset:offset + signal_length])
                    offset += stride
            else:
                split_indices = list(np.arange(0, len(signal_data[0]), signal_length))  # 10 seconds
                for idx in split_indices:
                    split_times.append(idx * signal_t)
                #split_times = split_times[:-1]

                split_sigs = np.split(signal_data, split_indices, axis=1)[1:-1]

            if len(split_sigs) < 1:
                print(f"Signal duration is too small: {fname}")
                continue
            split_times = split_times[1:]
            bpm_per_split, rt_per_split, pwa_per_split, area_per_split = calc_features(all_valley_times, all_peak_times,
                                                                                       all_valley_heights,
                                                                                       all_peak_heights, split_times,
                                                                                       gt_data, valleys, split_sigs, dataset)

            #sig_valley_idx, sig_peak_idx = gt_to_signal_peaks(signal_data[0], all_valley_times, all_peak_times)

            #sig_peak_heights = [signal_data[1, i] for i in sig_peak_idx]
            #sig_peak_times = [signal_data[0, i] for i in sig_peak_idx]
            #sig_valley_heights = [signal_data[1, i] for i in sig_valley_idx]
            #sig_valley_times = [signal_data[0, i] for i in sig_valley_idx]

            # Plot on ground truth data
            #plt.plot(gt_time[:5000], gt_data[:5000])
            #plt.plot(all_valley_times[:100], all_valley_heights[:100], 'go')
            #plt.plot(all_peak_times[:100], all_peak_heights[:100], 'ro')
            #plt.show()

            # Plot on traces signal
            #plt.plot(signal_data[0, :200], signal_data[1, :200])
            #plt.plot(sig_valley_times[:10], sig_valley_heights[:10], 'go')
            #plt.plot(sig_peak_times[:10], sig_peak_heights[:10], 'ro')
            #plt.show()

            # Split signal based on valleys
            # split_indices = [sig_valley_idx[i] for i in range(len(sig_valley_idx)) if i % signal_length == 0]
            #split_indices = list(np.arange(1, len(signal_data[0]), signal_length))   # 10 seconds
            #if len(split_indices) <= 1:
            #    print(f"Signal duration is too small: {fname}")
            #    continue

            #split_sigs = np.split(signal_data, split_indices, axis=1)[1:]
            #split_sigs = split_sigs[:-1]
            #split_times = [i * gt_t for i in split_indices]

            #bpms = calc_bpms_nopeaks(split_indices, signal_data, signal_length, fname)
            #bpms = calc_bpms_peaks(split_times, all_peak_times, fname, split_sigs)
            #rts_per_split = calc_risetime(all_valley_times, all_peak_times, signal_length, fname)
            #amp_per_split = calc_pwa(all_valley_times, all_peak_times, all_valley_heights, all_peak_heights, signal_length, fname)

            # Speedup augmentation
            #if p_nr not in test_nrs:
            #    for i, sig in enumerate(split_sigs):
            #        speedup = np.random.uniform(1, 2)
            #        if np.random.randint(0, 2):
            #            speedup = 1 / speedup
            #        bpms[i] /= speedup
            #        split_sigs[i] = np.multiply(split_sigs[i], speedup)

            # Save the resulting signals and hrs
            for i in range(len(bpm_per_split)):
                np.savetxt(f"{signal_save_path}{fname}_{str(i)}.npy", split_sigs[i])
                np.savetxt(f"{signal_save_path[:-1]}_hr/{fname}_{str(i)}.npy", [bpm_per_split[i]], delimiter=",")
                np.savetxt(f"{signal_save_path[:-1]}_rt/{fname}_{str(i)}.npy", [rt_per_split[i]], delimiter=",")
                np.savetxt(f"{signal_save_path[:-1]}_pwa/{fname}_{str(i)}.npy", [pwa_per_split[i]], delimiter=",")
                np.savetxt(f"{signal_save_path[:-1]}_area/{fname}_{str(i)}.npy", [area_per_split[i]], delimiter=",")
                #np.savetxt(signal_save_path + fname + "_" + str(i) + ".csv", split_sigs[i], delimiter=",")
                #np.savetxt(hr_save_path + fname + "_" + str(i) + ".csv", [bpms[i]], delimiter=',')

            highest = max(highest, max(signal_data[1]))
            lowest = min(lowest, min(signal_data[1]))
        else:
            if fname.split('_')[-1] != "source2":
                print(f"GT-file not found: {fname}")

    print(f"Highest: {str(highest)}, lowest: {str(lowest)}")

    #dts = np.asarray(dt)
    #q25, q75 = np.percentile(dts, [25, 75])
    #bin_width = 2 * (q75 - q25) * len(dts) ** (-1 / 3)
    #bins = round((dts.max() - dts.min()) / bin_width)
    #plt.hist(dts, bins=bins)
    #plt.savefig("/tudelft.net/staff-umbrella/StudentsCVlab/rsangers/thesis/plots/{config.DATASET}_dt.png")


def split_signal_noisy(signal_path, peaks_path, signal_save_path, hr_save_path, signal_length, delay):
    files = glob.glob(signal_path + "*.csv")
    print("Splitting signals for " + str(len(files)) + " files")

    for file in tqdm(files, total=len(files)):
        file = file.replace('\\', '/')
        fname = file.split('/')[-1].split(" ")[0] + ".csv"

        if os.path.isfile(peaks_path + fname): #FOR VIPL
            peak_data = np.loadtxt(peaks_path + fname, delimiter=',', skiprows=1) #FOR VIPL

            peaks = np.argwhere(peak_data[:, 2])
            peaks = np.reshape(peaks, (len(peaks)))
            peak_times = [peak_data[i, 0] for i in peaks]

            #peak_heights = [peak_data[i, 1] for i in peaks]
            #plt.plot(peak_data[:, 0], peak_data[:,1])
            #plt.plot(peak_times, peak_heights, 'o')
            #plt.show()

            signal_data = np.loadtxt(file, delimiter=',')

            split_times = peak_times[0:len(peak_times):signal_length]
            split_times = [x + delay for x in split_times]


            #if (len(peak_times) + 1) % signal_length != 0:
            #    split_times.remove(split_times[-1])

            signal_split_times = []
            signal_split_heights = []
            for i in range(len(split_times) - 1):
                output_times = []
                output_signal = []

                duration = split_times[i + 1] - split_times[i]
                hr = 1 / (duration / signal_length / 1000) * 60
                if hr < 30:
                    print(hr)
                    print(fname)

                for j, s in enumerate(signal_data[0]):
                    if s >= split_times[i] and s < split_times[i + 1]:
                        output_signal.append(signal_data[1, j])
                        output_times.append(signal_data[0, j])
                    elif s >= split_times[i + 1]:
                        signal_split_times.append(s)
                        signal_split_heights.append(signal_data[1, j])
                        break
                output = np.stack((output_times, output_signal))
                #print(hr)

                np.savetxt(signal_save_path + fname.split(".")[0] + "_" + str(i) + ".csv", output, delimiter=',')
                np.savetxt(hr_save_path + fname.split(".")[0] + "_" + str(i) + ".csv", [hr], delimiter=',')
            #plt.plot(signal_data[0], signal_data[1])
            #plt.plot(signal_split_times, signal_split_heights, "o")
            #plt.show()


        #plt.plot(signal_data[0], signal_data[1])
        #plt.plot(peak_times, peak_heights, '.')
        #plt.show()


def split_signal_time(signal_path, peaks_path, signal_save_path, hr_save_path, t):
    files = glob.glob(signal_path + "*.csv")
    print("Splitting signals for " + str(len(files)) + " files")

    for file in tqdm(files, total=len(files)):
        file = file.replace('\\', '/')
        fname = file.split('/')[-1].split(" ")[0] + ".csv"

        if os.path.isfile(peaks_path + fname): #FOR VIPL
        # if os.path.isfile(peaks_path + fname.split(".")[0] + " PPG.csv"):  # FOR PURE
            peak_data = np.loadtxt(peaks_path + fname, delimiter=',', skiprows=1) #FOR VIPL

            peaks = np.argwhere(peak_data[:, 2])
            peaks = np.reshape(peaks, (len(peaks)))
            peak_times = [peak_data[i, 0] for i in peaks]

            signal_data = np.loadtxt(file, delimiter=',')
            signal_duration = (signal_data[0,-1] - signal_data[0,0]) / 1000
            n_windows = int(signal_duration / t)
            split_times = [signal_data[0,0] + t * (i + 1) * 1000 for i in range(n_windows)]

            start = 0
            for i in range(len(split_times) - 1):
                if i != 0:
                    start = split_times[i-1]
                end = split_times[i]

                window_peaks = []
                for p in peak_times:
                    if p >= start and p < end:
                        window_peaks.append(p)

                if len(window_peaks) > 1:
                    bpm = 1 / ((window_peaks[-1] - window_peaks[0]) / (len(window_peaks) - 1) / 1000) * 60

                    if bpm < 30 or bpm > 90:
                        print(bpm)

                    output_times = []
                    output_signal = []

                    for j, s in enumerate(signal_data[0]):
                        if s >= start and s < end:
                            output_signal.append(signal_data[1, j])
                            output_times.append(signal_data[0, j])
                        elif s >= split_times[i + 1]:
                            break
                    output = np.stack((output_times, output_signal))
                    print(bpm)
                    np.savetxt(signal_save_path + fname.split(".")[0] + "_" + str(i) + ".csv", output, delimiter=',')
                    np.savetxt(hr_save_path + fname.split(".")[0] + "_" + str(i) + ".csv", [bpm], delimiter=',')
                else:
                    print(fname)


        #plt.plot(signal_data[0], signal_data[1])
        #plt.plot(peak_times, peak_heights, '.')
        #plt.show()


def show_peaks(signal_path, peaks_path):
    files = glob.glob(signal_path + "*.csv")

    for file in tqdm(files, total=len(files)):
        file = file.replace('\\', '/')
        fname = file.split('/')[-1].split(" ")[0] + ".csv"

        if os.path.isfile(peaks_path + fname):
            peak_data = np.loadtxt(peaks_path + fname, delimiter=',', skiprows=1)

            peaks = np.argwhere(peak_data[:, 2])
            peaks = np.reshape(peaks, (len(peaks)))
            peak_times = [peak_data[i, 0] for i in peaks]

            signal_data = np.loadtxt(file, delimiter=',')
            peak_heights = []
            for p in peak_times:
                for i in range(len(signal_data[0])-1):
                    t1 = signal_data[0, i]
                    t2 = signal_data[0, i+1]
                    if t1 <= p and t2 > p:
                        t_total = t2 - t1
                        peak_heights.append((p - t1) / t_total * signal_data[1, i] + (t2 - p) / t_total * signal_data[1, i+1])

            plt.plot(signal_data[0], signal_data[1])
            plt.plot(peak_times, peak_heights, 'o')
            plt.show()


            print('Total signal duration (s): '+str((signal_data[0,-1] - signal_data[0,0])/1000))
            print("Amount of peaks: "+str(len(peak_times)))


def correct_bpm(hr_save_path):
    files = glob.glob(hr_save_path + "*.csv")
    print("Correcting bpm for " + str(len(files)) + " files")

    for file in tqdm(files, total=len(files)):
        old_hr = np.loadtxt(file)
        new_hr = 1 / old_hr * 3600
        np.savetxt(file, [new_hr], delimiter=',')


def plot_bpms(hr_save_path):
    files = glob.glob(hr_save_path + "*.csv")
    hrs = []

    for file in tqdm(files, total=len(files)):
        hrs.append(np.loadtxt(file).item())
    hrs = np.asarray(hrs)

    q25, q75 = np.percentile(hrs, [25, 75])
    bin_width = 2 * (q75 - q25) * len(hrs) ** (-1 / 3)
    bins = round((hrs.max() - hrs.min()) / bin_width)
    plt.hist(hrs, bins=bins, density=True)
    plt.savefig('fivepeaks_hrs.png')
    plt.show()


if __name__ == '__main__':
    #test_bpmcalc()

    use_gt = False
    use_stride = False

    signal_path = config.TRACES_FILTERED_PATH
    # signal_path = config.ST_MAPS_PATH
    # signal_save_path = config.SPLIT_TRACES[:-1]+"_gt"
    signal_save_path = config.SPLIT_TRACES
    # signal_save_path = config.SPLIT_STMAPS

    dataset = config.DATASET
    if dataset == 'vicar':
        # gt_path = "/tudelft.net/staff-umbrella/StudentsCVlab/rsangers/VicarPPGBeyond_SpO2Alignment/Signals/"
        gt_path = "/tudelft.net/staff-bulk/ewi/insy/VisionLab/mbittner/VicarPPGBeyond/cleanedSignals/"
    else:
        gt_path = config.TARGET_SIGNAL_DIR

    # delay = 500
    # signal_path = "C:/Users/ruben/Documents/thesis/data/vipl/traces_filtered/"
    #gt_path = "C:/Users/ruben/Documents/thesis/data/pure/clean_hr/"
    # signal_path = "C:/Users/ruben/Documents/thesis/data/vicar/vicar_traces_filtered/"
    # gt_path = "C:/Users/ruben/Documents/thesis/data/vicar/vicar_gt/"
    # peaks_path = "C:/Users/ruben/Documents/thesis/data/vipl_signal/"
    # signal_save_path = "C:/Users/ruben/Documents/thesis/data/test_cwt/"
    # hr_save_path = "C:/Users/ruben/Documents/thesis/data/fiveseconds_hr/"

    # correct_bpm(hr_save_path)
    # plot_bpms(hr_save_path)
    # split_stmap(signal_path, gt_path, signal_save_path, dataset, use_stride=use_stride)
    # split_signal(signal_path, 10, signal_save_path, signal_save_path)
    split_signal2(signal_path, gt_path, signal_save_path, dataset, use_gt=use_gt, use_stride=use_stride)
    # split_signal_time(signal_path, peaks_path, signal_save_path, hr_save_path, t)
    # split_signal_noisy(signal_path, peaks_path, signal_save_path, hr_save_path, signal_length, delay)
    # show_peaks(signal_path, peaks_path)

