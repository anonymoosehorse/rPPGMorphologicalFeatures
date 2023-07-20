from pytorch_grad_cam import GradCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, FullGrad
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
from torchvision.models import resnet18
import torch
import numpy as np
import glob
import matplotlib.pyplot as plt
import seaborn as sns

from models.CWTNet import CWTNet
import config
from utils.dataset import DataLoaderCWTNet


if __name__ == '__main__':
    model = CWTNet()
    model_path = "C:/Users/ruben/Documents/thesis/checkpoint/running_model_3.410497524816057.pt"
    cwt_path = "C:/Users/ruben/Documents/thesis/cwt_data/"
    hr_path = "C:/Users/ruben/Documents/thesis/cwt_hr/"
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model.load_state_dict(torch.load(model_path)["model_state_dict"])
    model = model.eval().to(device)

    all_files = glob.glob(cwt_path + "i*.csv")
    cwt_files = []
    for file in all_files:
        file = file.replace('\\', '/')
        cwt_files.append(cwt_path + "_".join(file.split("/")[-1].split("_")[1:]))
    print(cwt_files)

    # Build Dataloaders
    cwt_set = DataLoaderCWTNet(cwt_files=cwt_files, target_signal_path=hr_path)
    data_loader = torch.utils.data.DataLoader(
        dataset=cwt_set,
        batch_size=1,
        num_workers=config.NUM_WORKERS,
        shuffle=False
    )

    cnn = model.cnn
    target_layers = [cnn[7][-1]]
    # cam = EigenCAM(model=model, target_layers=target_layers, use_cuda=True)
    cam = GradCAM(model=model, target_layers=target_layers, use_cuda=True)

    for data in data_loader:
        cwt = data["cwt_maps"].to(config.DEVICE).squeeze(0)
        print(cwt.shape)

        grayscale_cam = cam(input_tensor=cwt, my_target=data["target"].to(config.DEVICE))
        grayscale_cam = grayscale_cam[0, :]
        #print(grayscale_cam)
        #visualization = show_cam_on_image(cwt[0, 0], grayscale_cam, use_rgb=True)
        ax = sns.heatmap(grayscale_cam)
        plt.show()
        ax = sns.heatmap(cwt[0].cpu())
        plt.show()
        #ax = sns.heatmap(cwt[0, 1].cpu())
        #plt.show()


