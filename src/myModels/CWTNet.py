import numpy as np
from torch import nn
import torch
import torchvision.models as models
import torch.nn.functional as F

'''
Backbone CNN for CWTNet model is a RestNet-18
'''


class CWTNet(nn.Module):
    def __init__(self):
        super(CWTNet, self).__init__()

        # resnet o/p -> bs x 1000
        # self.resnet18 = resnet18(pretrained=False)
        resnet = models.resnet18(pretrained=False)
        resnet.conv1 = nn.Conv2d(2, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        modules = list(resnet.children())[:-1]

        self.cnn = nn.Sequential(*modules)
        self.dropout = nn.Dropout(p = 0.5)
        self.linear1 = nn.Linear(512, 1000)
        self.linear2 = nn.Linear(1000, 1)

    def forward(self, cwt_maps):
        # cwt_maps = cwt_maps.unsqueeze(0)
        x = self.cnn(cwt_maps)
        x = x.view(x.size(0), -1)

        x = self.linear1(x)
        x = self.dropout(x)
        x = self.linear2(x)

        return x

    def name(self):
        return "CWTNet"


if __name__ == '__main__':
    model = CWTNet()
    real = torch.tensor(np.loadtxt("C:/Users/ruben/Documents/thesis/data/vipl/split_cwt/r_p1_v1_source1_0.csv", delimiter=','))
    imag = torch.tensor(np.loadtxt("C:/Users/ruben/Documents/thesis/data/vipl/split_cwt/i_p1_v1_source1_0.csv", delimiter=','))

    # real = self.transform(real)
    # imag = self.transform(imag)
    real = F.interpolate(real.unsqueeze(0).unsqueeze(0), (224, 224)).squeeze(0).squeeze(0)
    imag = F.interpolate(imag.unsqueeze(0).unsqueeze(0), (224, 224)).squeeze(0).squeeze(0)
    cwt_maps = np.stack((real, imag))

    data = torch.unsqueeze(torch.tensor(cwt_maps, dtype=torch.float), 0)

    print(model(data))

