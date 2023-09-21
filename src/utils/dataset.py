# import albumentations
import torch
import numpy as np
from PIL import Image
from PIL import ImageFile
from torch.utils.data import Dataset

# import config
from constants import DatasetStats
from utils.file_io import read_target_data, get_hr_data, get_hr_data_stmaps, get_hr_data_filtered
import torch.nn.functional as F
import cv2

ImageFile.LOAD_TRUNCATED_IMAGES = True


class DataLoaderCWTNet(Dataset):
    def __init__(self, cwt_files, target_signal_path,device):
        self.cwt_files = cwt_files
        self.cwt_path = self.cwt_files[0].parent
        #self.transform = transforms.Compose([
        #    transforms.ToPILImage(),
        #    transforms.Resize((224, 224)),
        #    transforms.ToTensor()
        #])
        self.target_path = target_signal_path
        self.device = device

    def __len__(self):
        return len(self.cwt_files)

    def __getitem__(self, index):
        file_name = self.cwt_files[index].stem

        real = torch.tensor(np.loadtxt(self.cwt_path / f"r_{file_name}.csv" , delimiter=','))
        imag = torch.tensor(np.loadtxt(self.cwt_path / f"i_{file_name}.csv", delimiter=','))

        #real = self.transform(real)
        #imag = self.transform(imag)
        real = F.interpolate(real.unsqueeze(0).unsqueeze(0), (224, 224)).squeeze(0).squeeze(0)
        imag = F.interpolate(imag.unsqueeze(0).unsqueeze(0), (224, 224)).squeeze(0).squeeze(0)
        cwt_maps = np.stack((real, imag))

        # target_hr = get_hr_data_cwt(file_name)
        target_hr = get_hr_data_filtered(file_name, self.target_path)

        return {
            "data": torch.tensor(cwt_maps, dtype=torch.float).to(self.device),
            "target": torch.tensor(target_hr, dtype=torch.float).to(self.device)
        }


class DataLoader1D(Dataset):
    def __init__(self, data_files, target_signal_path,device,cfg):
        self.data_files = data_files
        self.data_path = self.data_files[0].parent
        #self.transform = transforms.Compose([
        #    transforms.ToPILImage(),
        #    transforms.Resize((224, 224)),
        #    transforms.ToTensor()
        #])
        self.target_path = target_signal_path
        self.device = device
        self.cfg = cfg
        self.dataset_stats = DatasetStats(cfg)

    def __len__(self):
        return len(self.data_files)

    def __getitem__(self, index):
        file_name = self.data_files[index].stem

        #data = torch.tensor(np.loadtxt(self.data_path + file_name + ".csv", delimiter=','))
        data = torch.tensor(np.loadtxt(self.data_path / f"{file_name}.npy"))[1,:]

        target_hr = get_hr_data_filtered(file_name, self.target_path)

        # Normalize the input
        if self.cfg.model.name == 'tranformer1d':
            if self.cfg.dataset.name == "vipl":
                dif = self.dataset_stats.VIPL_MAX - self.dataset_stats.VIPL_MIN
                data = (data - self.dataset_stats.VIPL_MIN) / dif # Change to range 0-1
            elif self.cfg.dataset.name  == "vicar":
                dif = self.dataset_stats.VICAR_MAX - self.dataset_stats.VICAR_MIN
                data = (data - self.dataset_stats.VICAR_MIN) / dif

        return {
            "data": data.float().to(self.device),
            "target": torch.tensor(target_hr).float().to(self.device)
        }


class DataLoaderSTMaps(Dataset):
    def __init__(self, data_files, target_signal_path,device,cfg):
        self.data_files = data_files
        self.data_path = self.data_files[0].parent
        self.target_path = target_signal_path
        self.device = device
        self.cfg = cfg

    def __len__(self):
        return len(self.data_files)

    def __getitem__(self, index):
        file_name = self.data_files[index].stem

        if self.cfg.dataset.use_yuv == True:            
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

        if self.cfg.model.name == 'transformer2d':
            img_yuv = torch.permute(img_yuv, (2, 0, 1))
            img_yuv = F.interpolate(img_yuv, size=224, mode='linear')
            img_yuv = torch.permute(img_yuv, (2, 1, 0))
            if self.cfg.dataset.name == 'vipl':
                img_yuv = img_yuv[:, 8:-8, :]
            else:
                img_yuv = img_yuv[:, 20:-20, :]

        target_hr = get_hr_data_stmaps(file_name, self.target_path)

        return {
            "data": torch.permute(img_yuv, (2, 0, 1)).to(self.device),
            "target": torch.tensor(target_hr, dtype=torch.float).to(self.device)
        }
