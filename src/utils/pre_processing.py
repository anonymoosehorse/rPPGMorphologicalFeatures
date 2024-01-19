import numpy as np
from .signal_processing import pos,butter_lowpass_filter,detect_peaks,signal_to_cwt
import json
import h5py
import torch.nn.functional as F
from torchvision.transforms import ToTensor
from tqdm import tqdm
import torch
import pandas as pd


def scale_to_range(value,old_min,old_max,new_min=0,new_max=1):
    return ((new_max - new_min)*(value-old_min)/(old_max-old_min)) + new_min

def get_data_from_json(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)

    r = np.array(data["R"], dtype=float)
    g = np.array(data["G"], dtype=float)
    b = np.array(data["B"], dtype=float)
    t = np.array(data["Times"], dtype=float)
    return t,r,g,b

def pos_and_filter(r,g,b,fps):
    pos_signal = pos(r, g, b)
    np.nan_to_num(pos_signal, copy=False)
    pos_signal = butter_lowpass_filter(pos_signal, fps, 2,[0.5,6])
    return pos_signal

def read_and_process(traces_path,fps):
    traces_lookup = {}
    for traces_file in traces_path.glob("*.json"):
        t,r,g,b = get_data_from_json(traces_file)
        pos_signal = pos_and_filter(r,g,b,fps)
        name = traces_file.stem.split(' ')[0]
        traces_lookup[name] = np.array([t,pos_signal])
    return traces_lookup

def read_h5(h5_path):
    with h5py.File(h5_path, "r") as f:
        gt_data = np.array(f['data']['PPG'])
    return gt_data

def read_csv(csv_path):
    gt_data = pd.read_csv(csv_path)
    gt_data = gt_data['Signal'].to_numpy()
    return gt_data

def read_pure_json(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    gt_data = [data_point['Value']['waveform'] for data_point in data['/FullPackage']]
    gt_data = np.array(gt_data, dtype=float)
    return gt_data

def read_numpy(np_path):
    gt_data = np.loadtxt(
        np_path, delimiter=',', skiprows=1)[:, 1]
    return gt_data

def read_gt_data(gt_path,gt_fps,load_func):
    gt_lookup = {}
    
    for gt_file_path in gt_path.glob("*.*"):
        gt_data = load_func(gt_file_path)
        gt_t = 1000 / gt_fps
        gt_time = [i * gt_t for i in range(len(gt_data))]
        gt_lookup[gt_file_path.stem] = np.array([gt_time,gt_data])
    return gt_lookup

def linear_resample(signals,source_fps,new_fps):
    signals = ToTensor()(signals)
    signals = F.interpolate(signals, size=int(len(signals[0, 0]) * new_fps / source_fps), mode='linear')
    signals = signals.numpy()[0]
    return signals

def resample_data(signal_dict,source_fps,new_fps=30):
    resampled_dict = {}
    for key,value in signal_dict.items():
        resampled_dict[key] = linear_resample(value,source_fps,new_fps)
    return resampled_dict

def detect_extrema_to_dict(signal_dict):
    extrema_dict = {}
    for key,value in signal_dict.items():
        peak_idcs,valley_idcs = detect_peaks(value[1,:],0.3)
        extrema_dict[key] = {}
        extrema_dict[key]['peaks'] = peak_idcs
        extrema_dict[key]['valleys'] = valley_idcs
        
    return extrema_dict

def np_between(array,a,b):
    if not isinstance(array,np.ndarray):
        array = np.array(array)
    return np.logical_and(array >= a, array < b)

def torch_between(array,a,b):
    if not isinstance(array,torch.Tensor):
        array = torch.tensor(array)
    return torch.logical_and(array >= a, array < b)


def get_wave_properties(signal, signal_times,peak_idcs,valley_idcs,fps):

    properties_dict = {"PeakIndex":[],"RT":[],"PWA":[],"AUP":[],"HR":[]}
    for j in range(1,len(valley_idcs)):
        wave_start = valley_idcs[j-1]
        wave_end = valley_idcs[j]
        wave_peaks = peak_idcs[np_between(peak_idcs,wave_start,wave_end)]
        rise_time = 1000 * ((wave_peaks[0] - wave_start) / fps)
        pwa = signal[wave_peaks[0]] - np.mean([signal[wave_start],signal[wave_end]])
        # area = np.sum(signal[wave_start:wave_end]) / (signal_times[wave_end] - signal_times[wave_start])
        area = np.trapz(signal[wave_start:wave_end],signal_times[wave_start:wave_end])
        hr = 60_000 / (signal_times[wave_end] - signal_times[wave_start])
        properties_dict['PeakIndex'].append(wave_peaks[0])
        properties_dict['RT'].append(rise_time)
        properties_dict['PWA'].append(pwa)
        properties_dict['AUP'].append(area)
        properties_dict['HR'].append(hr)
    return properties_dict

def get_wave_properties_torch(signal, signal_times,peak_idcs,valley_idcs,fps):

    properties_dict = {"PeakIndex":[],"RT":[],"PWA":[],"AUP":[],"HR":[]}
    for j in range(1,len(valley_idcs)):
        wave_start = valley_idcs[j-1]
        wave_end = valley_idcs[j]
        wave_peaks = peak_idcs[torch_between(peak_idcs,wave_start,wave_end)]
        rise_time = 1000 * ((wave_peaks[0] - wave_start) / fps)
        pwa = signal[wave_peaks[0]] - torch.mean(torch.tensor([signal[wave_start],signal[wave_end]]))
        # area = torch.sum(signal[wave_start:wave_end]) / (signal_times[wave_end] - signal_times[wave_start])
        area = torch.trapz(signal[wave_start:wave_end],signal_times[wave_start:wave_end])
        
        hr = 60_000 / (signal_times[wave_end] - signal_times[wave_start])
        properties_dict['PeakIndex'].append(wave_peaks[0])
        properties_dict['RT'].append(rise_time)
        properties_dict['PWA'].append(pwa)
        properties_dict['AUP'].append(area)
        properties_dict['HR'].append(hr)
    return properties_dict

def create_splits(signal_dict,gt_dict,extrema_dict,fps,gt_fps,window_time_s=10):

    matched_splits = {}
    for name, data in tqdm(signal_dict.items()):
        time = data[0,:]
        signal = data[1,:]

        if name not in gt_dict:
            print(f"Skipping {name} as no Ground Truth could be found")
            continue
        
        split_indices = np.arange(0, len(signal), window_time_s*fps)
        gt_split_indices = gt_split_indices = ((split_indices / fps) * gt_fps).astype(int)
        
        gt_time = gt_dict[name][0,:]
        
        gt_sig = gt_dict[name][1,:]
        peak_idcs = np.array(extrema_dict[name]['peaks'])
        valley_idcs = np.array(extrema_dict[name]['valleys'])

        avg_properties = {"SplitIndex":[],"RT":[],"PWA":[],"AUP":[],"HR":[]}
        for i in range(1,len(split_indices)):
            peak_idcs_win = peak_idcs[np.logical_and(peak_idcs >= gt_split_indices[i-1],peak_idcs < gt_split_indices[i])]
            valley_idcs_win = valley_idcs[np.logical_and(valley_idcs >= gt_split_indices[i-1],valley_idcs < gt_split_indices[i])]
            wave_properties = get_wave_properties(
                gt_sig,
                gt_time,
                peak_idcs_win,
                valley_idcs_win,
                gt_fps)
            hr = 60 / (np.mean(np.diff(peak_idcs_win)) / gt_fps)
            avg_properties['SplitIndex'].append(i - 1)
            avg_properties['RT'].append(np.mean(wave_properties['RT']))
            avg_properties['PWA'].append(np.mean(wave_properties['PWA']))
            avg_properties['AUP'].append(np.mean(wave_properties['AUP']))
            avg_properties['HR'].append(np.mean(wave_properties['HR']))
        
        ## Scale the time to seconds
        to_seconds_scale = round(fps * np.mean(np.diff(time)))
        time /= to_seconds_scale

        ## Split Signal and times, Remove empty and unfilled windows through slicing
        avg_properties['SplitData'] = np.split(signal, split_indices)[1:-1]
        avg_properties['SplitTime'] = np.split(time, split_indices)[1:-1]

        matched_splits[name] = avg_properties
        
    return matched_splits
    
def remove_faulty_splits(splits):
    for name,data in splits.items():        
        for idx in sorted(data['SplitIndex'],reverse=True):
            if np.isnan(data['HR'][idx]) or not np_between(data['HR'][idx],25,240) or not np_between(data['RT'][idx],50,1000):                
                print(f"Removed Window {idx} of Video {name} due to missing values / incorrect HR or RT")
                for key in data.keys():
                    data[key].pop(idx)
    return splits



def create_cwt(splits):
    cwt_dataset = {}
    for name,data in tqdm(splits.copy().items()):
        cwt_list = []
        for sig,time in zip(data.pop('SplitData'),data.pop('SplitTime')):
            cwt = signal_to_cwt(time,sig,output_size=256)
            cwt_list.append(cwt)
        cwt_dataset[name] = data
        cwt_dataset[name]['SplitData'] = cwt_list

    return cwt_dataset