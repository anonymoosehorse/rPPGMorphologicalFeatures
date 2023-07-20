import torch
from torch import nn, Tensor
import torch.nn.functional as F
import numpy as np
import config

class CNN1D(nn.Module):

    def __init__(self, norm_factor):
        super().__init__()
        self.conv1 = nn.Conv1d(1, 32, 5)
        self.conv2 = nn.Conv1d(32, 32, 5)
        self.conv3 = nn.Conv1d(32, 64, 5)
        self.conv4 = nn.Conv1d(64, 64, 5)
        self.conv5 = nn.Conv1d(64, 128, 5)
        self.conv6 = nn.Conv1d(128, 128, 5)
        self.conv7 = nn.Conv1d(128, 256, 5)
        self.conv8 = nn.Conv1d(256, 256, 5)
        self.conv9 = nn.Conv1d(256, 512, 5)
        self.conv10 = nn.Conv1d(512, 512, 5)

        self.linear1 = nn.Linear(2048, 128)
        self.linear2 = nn.Linear(128, 32)
        self.linear3 = nn.Linear(32, 1)

        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm1d(32)
        self.dropout = nn.Dropout(0.5)
        self.maxpool = nn.MaxPool1d(2)
        self.flatten = nn.Flatten()

        self.norm_factor = norm_factor


    def forward(self, x: Tensor) -> Tensor:
        x = x[1].unsqueeze(0).unsqueeze(0)
        x = F.interpolate(x, 256, mode='linear')
        x = self.bn(self.relu(self.conv1(x)))
        x = self.relu(self.conv2(x))
        x = self.maxpool(self.relu(self.conv3(x)))
        x = self.relu(self.conv4(x))
        x = self.maxpool(self.relu(self.conv5(x)))
        x = self.relu(self.conv6(x))
        x = self.maxpool(self.dropout(self.relu(self.conv7(x))))
        x = self.relu(self.conv8(x))
        x = self.dropout(self.maxpool(self.relu(self.conv9(x))))
        x = self.dropout(self.relu(self.conv10(x)))

        x = self.linear1(self.flatten(x))
        x = self.linear2(self.dropout(x))
        x = self.linear3(x).view((1))

        x *= self.norm_factor

        return x


if __name__ == '__main__':
    model = CNN1D()
    model.train()
    model.to(torch.device('cuda:0'))

    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    x = torch.tensor(np.loadtxt("C:/Users/ruben/Documents/thesis/data/vipl/split_traces/p1_v1_source1_0.csv", delimiter=','))
    x = x[1].unsqueeze(0).unsqueeze(0).type(torch.FloatTensor)
    x = x.to(torch.device('cuda:0'))

    #x = 2 * torch.rand(200,1,1) - 1
    #x = x * 100 + 100
    #x = x.type(torch.IntTensor)
    #print(x)

    for i in range(1000):
        output = model(x)
        print(output.item())
        loss = loss_fn(output, torch.Tensor([0.3]).to(torch.device('cuda:0')))
        loss.backward()
        optimizer.step()