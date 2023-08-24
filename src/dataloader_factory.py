import glob
import numpy as np
import re
from utils.dataset import DataLoaderSTMaps, DataLoaderCWTNet, DataLoader1D
from constants import DataPaths,DataFolds
import torch

# Needed in VIPL DATASET where each data item has a different number of frames/maps
def collate_fn(batch):
    batched_st_map, batched_targets = [], []
    # for data in batch:
    #     batched_st_map.append(data["st_maps"])
    #     batched_targets.append(data["target"])
    # # torch.stack(batched_output_per_clip, dim=0).transpose_(0, 1)
    return batch

def get_dataloaders(cfg,device):

    paths = DataPaths(cfg)
    folds = DataFolds(cfg)

    if cfg.dataset.name == "vipl":
        fold = folds.VIPL_FOLDS[cfg.dataset.fold_number]
        if cfg.model.data_dim == '2d':
            files = list(paths.DATA_PATH.glob("r*.csv"))
            all_files = [paths.DATA_PATH+file.stem[2:] for file in files]
            video_files_test = [paths.DATA_PATH+file.stem[2:] for file in files if int(file.stem.split("_")[1][1:]) in fold[0]]
            video_files_val = [paths.DATA_PATH+file.stem[2:] for file in files if int(file.stem.split("_")[1][1:]) in fold[1]]
        else:
            all_files = paths.DATA_PATH.glob("*.npy")
            video_files_test = [file for file in all_files if int(file.stem.split("_")[0][1:]) in fold[0]]
            video_files_val = [file for file in all_files if int(file.stem.split("_")[0][1:]) in fold[1]]
    else:
        fold = folds.VICAR_FOLDS[cfg.dataset.fold_number]
        if cfg.model.data_dim == '2d':
            files = list(paths.DATA_PATH.glob("r*.csv"))
            all_files = [paths.DATA_PATH / file.stem[2:] for file in files]
            video_files_test = [paths.DATA_PATH / file.stem[2:] for file in files if int(file.stem.split("_")[1]) in fold[0]]
            video_files_val = [paths.DATA_PATH / file.stem[2:] for file in files if int(file.stem.split("_")[1]) in fold[1]]
        else:
            all_files = list(paths.DATA_PATH.glob("*.npy"))
            video_files_test = [file for file in all_files if int(file.stem.split("_")[0]) in fold[0]]
            video_files_val = [file for file in all_files if int(file.stem.split("_")[0]) in fold[1]]
    video_files_train = [file for file in all_files if file not in video_files_test and file not in video_files_val]

    video_files_train = np.array(video_files_train)
    video_files_test = np.array(video_files_test)
    video_files_val = np.array(video_files_val)

    print(f"Trainset: {len(video_files_train)}, Testset: {len(video_files_test)}, Trainset: {len(video_files_val)}")


    # Build Dataloaders
    if cfg.model.data_dim == "3d":
        train_set = DataLoaderSTMaps(data_files=video_files_train, target_signal_path=paths.TARGET_PATH,device=device,cfg=cfg)
        test_set = DataLoaderSTMaps(data_files=video_files_test, target_signal_path=paths.TARGET_PATH,device=device,cfg=cfg)
        val_set = DataLoaderSTMaps(data_files=video_files_val, target_signal_path=paths.TARGET_PATH,device=device,cfg=cfg)
    elif cfg.model.data_dim == "2d":
        train_set = DataLoaderCWTNet(cwt_files=video_files_train, target_signal_path=paths.TARGET_PATH,device=device)
        test_set = DataLoaderCWTNet(cwt_files=video_files_test, target_signal_path=paths.TARGET_PATH,device=device)
        val_set = DataLoaderCWTNet(cwt_files=video_files_val, target_signal_path=paths.TARGET_PATH,device=device)
    elif cfg.model.data_dim == "1d":
        train_set = DataLoader1D(data_files=video_files_train, target_signal_path=paths.TARGET_PATH,device=device,cfg=cfg)
        test_set = DataLoader1D(data_files=video_files_test, target_signal_path=paths.TARGET_PATH,device=device,cfg=cfg)
        val_set = DataLoader1D(data_files=video_files_val, target_signal_path=paths.TARGET_PATH,device=device,cfg=cfg)

    train_loader = torch.utils.data.DataLoader(
        dataset=train_set,
        batch_size=cfg.train.batch_size,
        num_workers=cfg.dataset.num_workers,
        shuffle=False
    )
    # Check if all data exists
    #for item in train_loader:
    #    pass

    print('\nTrain DataLoader constructed successfully!')

    test_loader = torch.utils.data.DataLoader(
        dataset=test_set,
        batch_size=cfg.train.batch_size,
        num_workers=cfg.dataset.num_workers,
        shuffle=False
    )
    # Check if all data exists
    #for item in test_loader:
    #    pass
    print('\nEvaluation DataLoader constructed successfully!')

    val_loader = torch.utils.data.DataLoader(
        dataset=val_set,
        batch_size=cfg.train.batch_size,
        num_workers=cfg.dataset.num_workers,
        shuffle=False
    )

    return train_loader,test_loader,val_loader