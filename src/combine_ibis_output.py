import numpy as np
import pandas as pd
from pathlib import Path
from matplotlib import pyplot as plt
import h5py
from tqdm import tqdm
import cv2
import argparse
from omegaconf import OmegaConf

def get_vid_length(vid_path):
    """
    Get the number of frames in a video file

    Parameters:
    vid_path (str): The path to the video file.

    Returns:
    int: The number of the video in frames. Returns -1 if the video file cannot be opened.
    """
    cap = cv2.VideoCapture(vid_path)
    try:
        l = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    except:
        print(f"Cannot open video file {vid_path}")
        return -1
    cap.release()
    return l

def read_all_data_in_path(data_path):
    """
    Read all ibis data files in the given path.

    Args:
        data_path (Path): The path to the data files.

    Returns:
        tuple: A tuple containing two arrays, p and t.
            - p (ndarray or DataFrame): The stacked parent data arrays.
            - t (ndarray): The stacked traces data arrays.
    """
    p_list = []
    t_list = []

    for p_file in data_path.glob("parent_*.seg"):
        t_file = p_file.parent / p_file.name.replace("parent","traces")

        p_data = np.loadtxt(str(p_file))
        t_data = np.loadtxt(str(t_file))

        p_list.append(p_data)
        t_list.append(t_data)

    try:
        p = np.stack(p_list)
    except:
        p = pd.DataFrame(p_list).to_numpy()
        
    t = np.stack(t_list)

    return p,t


def main():
    """Combine single traces files created by the IBIS_temporal method into a single hdf.

    Options:
        -v, --verbose   : set verbose mode on
        --dataset             : name of the dataset (vicar or vipl)
        --ibis-datapath       : path to the ibis data
        --check-validity      : flag to check if IBIS traces match the length of the original videos
    """

    parser = argparse.ArgumentParser("IBIS data cleaner")
    parser.add_argument('-v', '--verbose', action='store_true', help="verbose mode on")
    parser.add_argument('--dataset', type=str, default='vicar',choice=('vicar','vipl'), help='Dataset to use (vicar or vipl)')
    parser.add_argument('--ibis-datapath', type=str, required=True, help="choose the path to ibis data")
    parser.add_argument('--check-validity', action='store_true', required=True, help="choose the path to ibis data")
    
    # Parse given arguments
    args = parser.parse_args()

    data_cfg = OmegaConf.load('x_dataset_config.yaml')
    data_cfg = data_cfg[args.dataset]

    #### Check paths

    ## Check if we have a path to the real dataset for checking length
    if data_cfg.dataset_path is None and args.check_validity:
        print("Please provide a path to the dataset")
        exit()

    ibis_data_path = Path(args.ibis_datapath)
    if not ibis_data_path.exists():
        raise FileExistsError(f"Cannot find path {ibis_data_path}")
    
    dataset_path = None
    if args.check_validity:
        dataset_path = Path(data_cfg.dataset_path)
        if not dataset_path.exists():
            raise FileExistsError(f"Cannot find path {dataset_path}")    

    if next(ibis_data_path.rglob("*" + data_cfg.video_suffix)) is None:
        print("Found no data files")
        exit()

    #### Create hdf file

    h5_outfile = Path.cwd() / f"{args.dataset_name}_ibis_combined.h5"
    if h5_outfile.exists():    
        hf = h5py.File(h5_outfile,'a')
    else:
        hf = h5py.File(h5_outfile,'w')

    ## Match names
    if args.check_validity:
        print("Generating video lookup table ...")
        video_lookup = {vid_path.name:vid_path for vid_path in tqdm(dataset_path.rglob("*" + data_cfg.video_suffix))}

    if args.verbose:
        print("Combining in hdf file")

    for out_folder_path in tqdm(ibis_data_path.rglob("*" + data_cfg.video_suffix)):        

        name = out_folder_path.stem
        
        if name in hf:
            if args.verbose:
                print(f"Skipping {name} as it is already in Folder...")
            continue
        
        ## Check if video and traces have the same length
        if args.check_validity:
            vid_length = get_vid_length(str(video_lookup[out_folder_path.name]))
            ibis_length = (len(list(out_folder_path.glob("traces_*.seg"))))

            if vid_length != ibis_length:
                print(f"{out_folder_path.stem} should have {vid_length} samples but has {ibis_length} samples, skipping!")
                continue
        
        if args.verbose:
            print(f"Processing output folder {out_folder_path}...")

        p,t = read_all_data_in_path(out_folder_path)
        grp = hf.create_group(name)
        grp.create_dataset('parent',data=p)
        grp.create_dataset('traces',data=t)
    hf.close()
    

if __name__=="__main__":
    main()
