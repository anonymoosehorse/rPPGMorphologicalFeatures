# import albumentations
import torch
import numpy as np
from PIL import Image
from PIL import ImageFile
from torch.utils.data import Dataset

import config
from utils.signal_utils import read_target_data, calculate_hr, get_hr_data, get_hr_data_stmaps, get_hr_data_filtered
import glob
from torchvision import transforms
import torch.nn.functional as F
import cv2
import re

ImageFile.LOAD_TRUNCATED_IMAGES = True


class DataLoaderCWTNet(Dataset):
    def __init__(self, cwt_files, target_signal_path):
        self.cwt_files = cwt_files
        self.cwt_path = '/'.join(re.split(r"\\|/",self.cwt_files[0])[:-1]) + "/"
        #self.transform = transforms.Compose([
        #    transforms.ToPILImage(),
        #    transforms.Resize((224, 224)),
        #    transforms.ToTensor()
        #])
        self.target_path = target_signal_path

    def __len__(self):
        return len(self.cwt_files)

    def __getitem__(self, index):
        file_name = ".".join(re.split(r"\\|/",self.cwt_files[index])[-1].split('.')[:-1])

        real = torch.tensor(np.loadtxt(self.cwt_path + "r_" + file_name + ".csv", delimiter=','))
        imag = torch.tensor(np.loadtxt(self.cwt_path + "i_" + file_name + ".csv", delimiter=','))

        #real = self.transform(real)
        #imag = self.transform(imag)
        real = F.interpolate(real.unsqueeze(0).unsqueeze(0), (224, 224)).squeeze(0).squeeze(0)
        imag = F.interpolate(imag.unsqueeze(0).unsqueeze(0), (224, 224)).squeeze(0).squeeze(0)
        cwt_maps = np.stack((real, imag))

        # target_hr = get_hr_data_cwt(file_name)
        target_hr = get_hr_data_filtered(file_name, self.target_path)

        return {
            "data": torch.tensor(cwt_maps, dtype=torch.float),
            "target": torch.tensor(target_hr, dtype=torch.float)
        }


class DataLoader1D(Dataset):
    def __init__(self, data_files, target_signal_path):
        self.data_files = data_files
        self.data_path = '/'.join(re.split(r"\\|/",self.data_files[0])[:-1]) + "/"
        #self.transform = transforms.Compose([
        #    transforms.ToPILImage(),
        #    transforms.Resize((224, 224)),
        #    transforms.ToTensor()
        #])
        self.target_path = target_signal_path

    def __len__(self):
        return len(self.data_files)

    def __getitem__(self, index):
        file_name = ".".join(re.split(r"\\|/",self.data_files[index])[-1].split('.')[:-1])

        #data = torch.tensor(np.loadtxt(self.data_path + file_name + ".csv", delimiter=','))
        data = torch.tensor(np.loadtxt(self.data_path + file_name + ".npy"))

        target_hr = get_hr_data_filtered(file_name, self.target_path)

        return {
            "data": torch.tensor(data, dtype=torch.float),
            "target": torch.tensor(target_hr, dtype=torch.float)
        }


class DataLoaderSTMaps(Dataset):
    def __init__(self, data_files, target_signal_path):
        self.data_files = data_files
        self.data_path = '/'.join(re.split(r"\\|/",self.data_files[0])[:-1]) + "/"
        self.target_path = target_signal_path

    def __len__(self):
        return len(self.data_files)

    def __getitem__(self, index):
        file_name = ".".join(re.split(r"\\|/",self.data_files[index])[-1].split('.')[:-1])

        if config.USE_YUV == True:
            img_yuv = np.load(self.data_path + file_name + ".npy")
            img_yuv = cv2.cvtColor(np.float32(img_yuv * 255), cv2.COLOR_BGR2YUV)
            img_yuv = torch.tensor(img_yuv, dtype=torch.float)
            img_yuv = torch.nan_to_num(img_yuv)
            img_yuv = torch.permute(img_yuv, (1, 0, 2))
        else:
            img_yuv = np.loadtxt(self.data_path + file_name + ".npy", delimiter=',')
            # img_yuv = np.load(self.data_path + file_name + ".npy")
            img_yuv = torch.tensor(img_yuv, dtype=torch.float)
            img_yuv = torch.unsqueeze(img_yuv, 2)

        if config.MODEL == 'transformer2d':
            img_yuv = torch.permute(img_yuv, (2, 0, 1))
            img_yuv = F.interpolate(img_yuv, size=224, mode='linear')
            img_yuv = torch.permute(img_yuv, (2, 1, 0))
            if config.DATASET == 'vipl':
                img_yuv = img_yuv[:, 8:-8, :]
            else:
                img_yuv = img_yuv[:, 20:-20, :]

        target_hr = get_hr_data_stmaps(file_name, self.target_path)

        return {
            "data": torch.permute(img_yuv, (2, 0, 1)),
            "target": torch.tensor(target_hr, dtype=torch.float)
        }


class DataLoaderRhythmNet(Dataset):
    """
        Dataset class for RhythmNet
    """

    # The data is now the SpatioTemporal Maps instead of videos

    def __init__(self, st_maps_path, target_signal_path):
        self.H = 180
        self.W = 180
        self.C = 3
        # self.video_path = data_path
        # self.st_maps_path = glob.glob(st_maps_path + '/*.npy')
        self.st_maps_path = st_maps_path
        # self.resize = resize
        self.target_path = target_signal_path
        self.maps = None

        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
        # Maybe add more augmentations
        # self.augmentation_pipeline = albumentations.Compose(
        #     [
        #         albumentations.Normalize(
        #             mean, std, max_pixel_value=255.0, always_apply=True
        #         )
        #     ]
        # )

    def __len__(self):
        return len(self.st_maps_path)

    def __getitem__(self, index):
        # identify the name of the video file so as to get the ground truth signal
        self.video_file_name = re.split(r"\\|/",self.st_maps_path[index])[-1].split('.')[0]
        # targets, timestamps = read_target_data(self.target_path, self.video_file_name)
        # sampling rate is video fps (check)

        # Load the maps for video at 'index'
        self.maps = np.load(self.st_maps_path[index])
        map_shape = self.maps.shape
        self.maps = self.maps.reshape((-1, map_shape[3], map_shape[1], map_shape[2]))

        # target_hr = calculate_hr(targets, timestamps=timestamps)
        # target_hr = calculate_hr_clip_wise(map_shape[0], targets, timestamps=timestamps)
        target_hr = get_hr_data(self.video_file_name)

        # To check the fact that we dont have number of targets greater than the number of maps
        # target_hr = target_hr[:map_shape[0]]
        self.maps = self.maps[:target_hr.shape[0], :, :, :]

        return {
            "st_maps": torch.tensor(self.maps, dtype=torch.float),
            "target": torch.tensor(target_hr, dtype=torch.float)
        }
