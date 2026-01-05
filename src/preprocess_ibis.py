import numpy as np
from matplotlib import pyplot as plt
from pathlib import Path,PurePath
import json
import pickle as pkl
from tqdm import tqdm
import h5py
from scipy import signal
from scipy.interpolate import interp1d
from omegaconf import OmegaConf
import argparse
from .utils.signal_processing import butter_lowpass_filter, pos_img, fir_bp_filter


def load_traces_data_to_dict(traces_data_path):

    traces_data = {}
    with h5py.File(traces_data_path,'r') as f:
        for name,group in f.items():
            traces_data[name] = {}
            for sub_name,data in group.items():
                traces_data[name][sub_name] = data[:]

    return traces_data

def build_ibis_tracks(traces_data, parent_data):
    coherent_traces = traces_data.copy()
    n_frames, n_traces, _ = traces_data.shape

    for i_f in range(n_frames):
        for i_t in range(n_traces):
            if np.isnan(parent_data[i_f, i_t]) or int(parent_data[i_f, i_t]) == i_t or i_f == 0:
                coherent_traces[i_f, i_t] = traces_data[i_f, i_t]
            else:
                coherent_traces[:i_f, i_t] = coherent_traces[:i_f, int(parent_data[i_f, i_t])]

    return coherent_traces

def super_custom_resampling_vipl(dataset_path,file_name,traces_data_path,data,traces_fps):

        """
        Custom function in case the trace data for VIPL was generated from videos
        that were already resampled to match the timestamps in the time.txt files
        """
        
        success = False

        json_data_path = Path(traces_data_path) / f"{file_name} SP.json"

        with open(json_data_path,'r') as f:
            json_data = json.load(f)

        time_data_path = Path(PurePath(str(dataset_path), *file_name.split("_"), "time.txt"))

        if not time_data_path.exists():
            success = True
            return success,data

        time_data = np.loadtxt(time_data_path) / 1000
        if len(time_data) != len(data):            
            return success,data

        it = np.arange(0,time_data[-1],1/traces_fps)
        it = it[:len(json_data['R'])]
        interp_data = np.zeros([len(it),*data.shape[1:]])
        for row_idx in range(data.shape[1]):            
                f = interp1d(time_data,data[:,row_idx])
                interp_data[:,row_idx] = f(it)        

        data = interp_data
        del interp_data  

        success = True 
        return success,data

def resample_vipl_to_constant_fps(dataset_path,file_name,data,cfg,traces_fps):
        
        success = False

        time_data_path = Path(PurePath(str(dataset_path), *file_name.split("_"), "time.txt"))

        if not time_data_path.exists():
            success = True
            return success,data

        time_data = np.loadtxt(time_data_path) / 1000
        
        if len(time_data) != len(data):            
            return success,data

        it = np.arange(0,time_data[-1],1/traces_fps)
        
        interp_data = np.zeros([len(it),*data.shape[1:]])
        for row_idx in range(data.shape[1]):
            for channel_idx in range(data.shape[2]):
                f = interp1d(time_data,data[:,row_idx,channel_idx])
                interp_data[:,row_idx,channel_idx] = f(it)        

        data = interp_data
        del interp_data 

        success = True 
        return success,data

def preprocess_ibis(cfg):
    """ Match the IBIS data to the traces data / Preprocess IBIS data

    Options:
        
        --dataset               : name of the dataset (vicar or vipl)
        --ibis-path             : path to the ibis hdf file created by combine_ibis_output.py    
    """

    # parser = argparse.ArgumentParser(description='Convert IBIS data to h5 format')
    # parser.add_argument('--dataset', type=str, default='vicar',choices=('vicar','vipl','pure','ubfc1','ubfc2','ucla'), help='Dataset to use (vicar or vipl)')
    # parser.add_argument('--ibis-path',default="", type=str, help='Path to the IBIS hdf file')    

    # args = parser.parse_args()

    print(cfg)

    traces_hdf_path = Path(cfg.output_directory) / cfg.dataset_to_run / "traces_data.h5"
    traces_data = load_traces_data_to_dict(traces_hdf_path)

    ibis_path = Path(cfg.ibis_hdf_path)    

    if not ibis_path.exists():
        print(f"Cannot find path {ibis_path}")
        exit()

    ibis_data_path = traces_hdf_path.parent / "ibis_data.h5"
    hf_in = h5py.File(ibis_path,'r')

    hf =  h5py.File(ibis_data_path, 'w')

    for name,group in tqdm(hf_in.items()):
        
        data = build_ibis_tracks(group['traces'][:],group['parent'][:])
        data = pos_img(data,axis=0)
        data = fir_bp_filter(data, cfg.traces_fps, 51,axis=0)

        file_name = name    

        if file_name not in traces_data:
            print(f"Skipping {file_name} as it probably does not have a GroundTruth")        
            continue

        if cfg.dataset_to_run == 'vipl':
            ## Some VIPL videos come with provided time stamps
            ## We can use these to resample the data to a constant frame rate

            # success, new_data = resample_vipl_to_constant_fps(
            #     cfg.dataset_path,
            #     file_name,                
            #     data,
            #     cfg.traces_fps)
            success, new_data = super_custom_resampling_vipl(
                cfg.dataset_path,
                file_name,
                cfg.traces_path,
                data,
                cfg.traces_fps)
            if not success:
                print(f"Something going wrong with IBIS data of file {file_name}")
                del traces_data[file_name]
                continue
            else:
                data = new_data
                del new_data
            
        ## Resample the data to fit 30 fps
        if cfg.traces_fps != 30:        
            data = signal.resample(data,len(np.arange(0,data.shape[0]/cfg.traces_fps,1/30)),axis=0)
    
        split_indices = np.arange(0, len(data), 10*30)
        split_images = np.split(data, split_indices)[1:-1]

        if len(split_images) != len(traces_data[file_name]['SplitIndex']):    
            
            additional_values = set(range(len(split_images))).difference(set(traces_data[file_name]['SplitIndex']))
            for i in sorted(additional_values,reverse=True):
                print(f"Removing instance for Index: {i} for {file_name}")
                split_images.pop(i)

        if len(split_images) != len(traces_data[file_name]['SplitIndex']): 
            not_contained_ids = set(range(len(traces_data[file_name]['SplitIndex']))).difference(set(range(len(split_images))))
            for key in traces_data[file_name].keys():
                for idx in sorted(not_contained_ids,reverse=True):
                    if isinstance(traces_data[file_name][key],np.ndarray):
                        np.delete(traces_data[file_name][key],idx)
                    elif isinstance(traces_data[file_name][key],list):
                        traces_data[file_name][key].pop(idx)
                    print(f"Removed Split Index {idx} as it had no corresponding traces")

        # traces_data[file_name]['SplitData'] = split_images    

        group = hf.create_group(file_name)
        for key,sub_splits in traces_data[file_name].items():
            if key == 'SplitData':
                group.create_dataset(key,data=split_images)
            else:
                group.create_dataset(key,data=sub_splits)


    hf.close()

if __name__=="__main__":

    
    cli  = OmegaConf.from_cli()
    prep_cfg = OmegaConf.load('x_preprocess_config.yaml')
    cfg = OmegaConf.load('x_dataset_config.yaml')
    cfg = OmegaConf.merge(cfg[prep_cfg.dataset_to_run], prep_cfg, cli)

    preprocess_ibis(cfg)


    print("Done")