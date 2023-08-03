from myModels.CWTNet import CWTNet
from myModels.CWTNet2 import CWTNet2
from myModels.TransformerModel import TransformerModel, NoamOpt
from myModels.CNN1D import CNN1D
from myModels.resnet1d import ResNet1D
from myModels.resnet2d import Resnet2D


def get_model(model_name: str,data_dimensions,norm_factor = 1):


    if model_name == 'resnet1d':
        model = ResNet1D(
            in_channels=1,
            base_filters=64,  # 64 for ResNet1D, 352 for ResNeXt1D
            kernel_size=16,
            stride=2,
            groups=1,
            n_block=10,
            n_classes=1,
            norm_factor=norm_factor,
            downsample_gap=2,
            increasefilter_gap=2,
            use_do=True,
            use_bn=True)
    elif model_name == 'resnet2d':
        model = Resnet2D(data_dimensions)
    elif model_name == 'transformer1d':
        model = TransformerModel(seq_len=300, d_model=256, nhead=4, d_hid=2048, nlayers=8,
                                 norm_factor=norm_factor)
    elif model_name == 'transformer2d':
        model = CWTNet2(model_name, data_dimensions)
    else:
        print("Invalid model type")

    return model