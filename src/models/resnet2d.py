import torch
from torch import nn
from torch.nn import functional as F
import torchvision.models as models
import ssl
# import config
import numpy as np
import cv2
from PIL import Image

ssl._create_default_https_context = ssl._create_stdlib_context

'''
Backbone CNN for RhythmNet model is a RestNet-18
'''


class Resnet2D(nn.Module):
    def __init__(self, data_dim,output_dim=1,use_yuv=False):
        super(Resnet2D, self).__init__()

        resnet = models.resnet18(pretrained=False)
        if data_dim == '2d':
            resnet.conv1 = nn.Conv2d(2, 64, kernel_size=7, stride=2, padding=3,
                                    bias=False)
        elif data_dim == '3d':
            if use_yuv:
                resnet.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                                         bias=False)
            else:
                resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3,
                                         bias=False)
        modules = list(resnet.children())[:-1]
        self.resnet18 = nn.Sequential(*modules)

        self.dropout = nn.Dropout(p=0.5)
        self.fc1 = nn.Linear(512, 1000)
        self.fc2 = nn.Linear(1000, output_dim)

    def forward(self, x):                
        x = self.resnet18(x)
        x = x.view(x.size(0), -1)

        x = self.fc1(x)
        x = self.dropout(x)
        x = self.fc2(x) 

        return x


if __name__ == '__main__':
    # cm = RhythmNet()
    # img = torch.rand(3, 28, 28)
    # target = torch.randint(1, 20, (5, 5))
    # x = cm(img)
    # print(x)
    x = np.load("C://Users//ruben//Documents//thesis//data//vipl//split_stmaps//p1_v1_s1_0.npy")
    #x = np.transpose(x, (2, 0, 1))
    print(x.shape)
    img_yuv = cv2.cvtColor(np.float32(x * 255), cv2.COLOR_BGR2YUV)
    x = torch.tensor(img_yuv)
    x = torch.nan_to_num(x)

    x = torch.unsqueeze(x, 0)
    x = torch.permute(x, (0, 3, 1, 2))
    resnet18 = Resnet2D()
    y = resnet18(x)
    print(y)
