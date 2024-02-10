from torch import nn
import torch
import numpy as np
import torch.nn.functional as F

# import config

'''
Backbone CNN for CWTNet model is a DeIT
'''


class CWTNet2(nn.Module):
    def __init__(self, model, data_dim,output_dim,use_yuv=False):
        super(CWTNet2, self).__init__()
        self.transformer = torch.hub.load('facebookresearch/deit:main', 'deit_base_patch16_224', pretrained=True, source='github')
        #else:
        #    self.transformer = torch.hub.load('facebookresearch/deit:main', 'deit_small_patch16_224', pretrained=True, source='github')

        if data_dim == '2d':
            self.transformer.patch_embed.proj = nn.Conv2d(2, 768, kernel_size=(16, 16), stride=(16, 16))
        elif data_dim == '3d':
            if use_yuv:
                self.transformer.patch_embed.proj = nn.Conv2d(3, 768, kernel_size=(16, 16), stride=(16, 16))
            else:
                self.transformer.patch_embed.proj = nn.Conv2d(1, 768, kernel_size=(16, 16), stride=(16, 16))

        self.transformer.head = nn.Linear(self.transformer.head.in_features, 1000)
        self.dropout = nn.Dropout(0.5)
        self.linear = nn.Linear(1000, output_dim)


    def forward(self, x):
        #x = x.unsqueeze(0)

        x = self.transformer(x)
        x = self.dropout(x)
        x = self.linear(x)

        return x

    def name(self):
        return "CWTNet"


if __name__ == '__main__':
    model = CWTNet2('deit_base', '3d')
    data = torch.tensor(np.load("C:/Users/ruben/Documents/thesis/data/vipl/split_stmaps/p1_v1_s1_0.npy"))
    print(data.shape)
    data = torch.permute(data, (2, 1, 0))
    data = F.interpolate(data, size=int(224), mode='linear').type(torch.FloatTensor)
    x = torch.unsqueeze(data, 0)
    x = x[:, :, 8:-8, :]
    print(x.shape)

    y = model(x)
