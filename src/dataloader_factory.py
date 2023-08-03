import glob
import numpy as np
import re
from utils.dataset import DataLoaderSTMaps, DataLoaderCWTNet, DataLoader1D
import torch

# Needed in VIPL DATASET where each data item has a different number of frames/maps
def collate_fn(batch):
    batched_st_map, batched_targets = [], []
    # for data in batch:
    #     batched_st_map.append(data["st_maps"])
    #     batched_targets.append(data["target"])
    # # torch.stack(batched_output_per_clip, dim=0).transpose_(0, 1)
    return batch

def get_dataloaders(config,device):

    if config.DATASET == "vipl":
        fold = config.VIPL_FOLDS[config.FOLD_NR]
        if config.DATA_DIM == '2d':
            files = list(config.DATA_PATH.glob("r*.csv"))
            all_files = [config.DATA_PATH+file.stem[2:] for file in files]
            video_files_test = [config.DATA_PATH+file.stem[2:] for file in files if int(file.stem.split("_")[1][1:]) in fold[0]]
            video_files_val = [config.DATA_PATH+file.stem[2:] for file in files if int(file.stem.split("_")[1][1:]) in fold[1]]
        else:
            all_files = config.DATA_PATH.glob("*.npy")
            video_files_test = [file for file in all_files if int(file.stem.split("_")[0][1:]) in fold[0]]
            video_files_val = [file for file in all_files if int(file.stem.split("_")[0][1:]) in fold[1]]
    else:
        fold = config.VICAR_FOLDS[config.FOLD_NR]
        if config.DATA_DIM == '2d':
            files = list(config.DATA_PATH.glob("r*.csv"))
            all_files = [config.DATA_PATH+file.stem[2:] for file in files]
            video_files_test = [config.DATA_PATH+file.stem[2:] for file in files if int(file.stem.split("_")[1]) in fold[0]]
            video_files_val = [config.DATA_PATH+file.stem[2:] for file in files if int(file.stem.split("_")[1]) in fold[1]]
        else:
            all_files = list(config.DATA_PATH.glob("*.npy"))
            video_files_test = [file for file in all_files if int(file.stem.split("_")[0]) in fold[0]]
            video_files_val = [file for file in all_files if int(file.stem.split("_")[0]) in fold[1]]
    video_files_train = [file for file in all_files if file not in video_files_test and file not in video_files_val]

    video_files_train = np.array(video_files_train)
    video_files_test = np.array(video_files_test)
    video_files_val = np.array(video_files_val)
    print(f"Trainset: {len(video_files_train)}, Testset: {len(video_files_test)}, Trainset: {len(video_files_val)}")


    # Build Dataloaders
    if config.DATA_DIM == "3d":
        train_set = DataLoaderSTMaps(data_files=video_files_train, target_signal_path=config.TARGET_PATH,device=device)
        test_set = DataLoaderSTMaps(data_files=video_files_test, target_signal_path=config.TARGET_PATH,device=device)
        val_set = DataLoaderSTMaps(data_files=video_files_val, target_signal_path=config.TARGET_PATH,device=device)
    elif config.DATA_DIM == "2d":
        train_set = DataLoaderCWTNet(cwt_files=video_files_train, target_signal_path=config.TARGET_PATH,device=device)
        test_set = DataLoaderCWTNet(cwt_files=video_files_test, target_signal_path=config.TARGET_PATH,device=device)
        val_set = DataLoaderCWTNet(cwt_files=video_files_val, target_signal_path=config.TARGET_PATH,device=device)
    elif config.DATA_DIM == "1d":
        train_set = DataLoader1D(data_files=video_files_train, target_signal_path=config.TARGET_PATH,device=device)
        test_set = DataLoader1D(data_files=video_files_test, target_signal_path=config.TARGET_PATH,device=device)
        val_set = DataLoader1D(data_files=video_files_val, target_signal_path=config.TARGET_PATH,device=device)

    train_loader = torch.utils.data.DataLoader(
        dataset=train_set,
        batch_size=config.BATCH_SIZE,
        num_workers=config.NUM_WORKERS,
        shuffle=False,
        collate_fn=collate_fn
    )
    # Check if all data exists
    #for item in train_loader:
    #    pass

    print('\nTrain DataLoader constructed successfully!')

    test_loader = torch.utils.data.DataLoader(
        dataset=test_set,
        batch_size=config.BATCH_SIZE,
        num_workers=config.NUM_WORKERS,
        shuffle=False,
        collate_fn=collate_fn
    )
    # Check if all data exists
    #for item in test_loader:
    #    pass
    print('\nEvaluation DataLoader constructed successfully!')

    val_loader = torch.utils.data.DataLoader(
        dataset=val_set,
        batch_size=config.BATCH_SIZE,
        num_workers=config.NUM_WORKERS,
        shuffle=False,
        collate_fn=collate_fn
    )

    return train_loader,test_loader,val_loader