import math
from typing import Tuple

import numpy as np

import config

import torch
from torch import nn, Tensor
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.utils.data import dataset

class TransformerModel(nn.Module):

    def __init__(self, seq_len: int, d_model: int, nhead: int, d_hid: int,
                 nlayers: int, norm_factor, dropout: float = 0.1):
        super().__init__()
        self.norm_factor = norm_factor

        self.pos_encoder = PositionalEncoding(d_model, dropout)
        self.layer_norm = nn.LayerNorm(d_model)
        encoder_layers = TransformerEncoderLayer(d_model, nhead, d_hid, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers, self.layer_norm)
        # self.decoder2 = nn.Linear(64, 1)
        # self.ntoken = ntoken
        self.seq_len = seq_len

        self.device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
        self.pos = torch.unsqueeze(torch.tensor(np.linspace(0, 1, seq_len)), 0).to(self.device)

        # self.encoder = nn.Embedding(ntoken, d_model)
        # self.d_model = d_model

        # self.decoder = nn.Linear(seq_len, 64)
        self.decoder = nn.Linear(seq_len, 1)
        self.dropout = nn.Dropout(dropout)
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        # self.self_attn_pool = SelfAttentionPooling(d_model)

        # self.conv1 = torch.nn.Conv1d(in_channels=1, out_channels=128, kernel_size=3, stride=1, padding=1)
        # self.conv2 = torch.nn.Conv1d(in_channels=128, out_channels=d_model, kernel_size=3, stride=1, padding=1)
        # self.conv = torch.nn.Conv1d(in_channels=d_model, out_channels=d_model, kernel_size=3, stride=1, padding=1)
        self.conv1 = torch.nn.Conv1d(in_channels=1, out_channels=12, kernel_size=3, stride=1, padding=1)
        self.conv2 = torch.nn.Conv1d(in_channels=12, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.conv3 = torch.nn.Conv1d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.conv4 = torch.nn.Conv1d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.relu = torch.nn.ReLU()
        # self.maxpool = torch.nn.MaxPool1d(kernel_size=2)

        #self.init_weights()

    def init_weights(self) -> None:
        initrange = 0.1
        # self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)
        #self.decoder2.bias.data.zero_()
        #self.decoder2.weight.data.uniform_(-initrange, initrange)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor, shape [batch_size, 2, seq_len]

        Returns:
            output Tensor of shape [seq_len, batch_size, ntoken]
        """
        # Remove time axis
        # x = x[:,1,:] # Shape(batch_size, seq_len)

        # * self.ntoken
        # Quantize the input and add positional encoding as extra dimension
        #x = torch.floor(x * self.ntoken)  # Change to range 0-nToken
        #x = x / self.ntoken  # Change to quantized range 0-1
        #pos = self.pos.expand(x.size(dim=0), -1)
        #x = torch.stack((x, pos))  # Shape(2, batch_size, seq_len)
        #x = torch.permute(x, (2, 1, 0)).type(torch.FloatTensor).to(self.device) # Shape(seq_len, batch_size, 2)

        x = x.unsqueeze(1)
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.relu(self.conv4(x))

        #for i in range(2):
        #    x = self.relu(self.conv(x))
        #    x = self.maxpool(x)
        x = torch.permute(x, (2, 0, 1))
        x = self.pos_encoder(x)
        #x = torch.permute(x, (2, 0, 1))

        # Generate attention mask
        x = self.transformer_encoder(x)
        #x = torch.permute(x, (1, 2, 0))  # Resize to --> [batch, seq_len, 1]

        # Perform average pooling over the token dimension
        x = torch.permute(x, (1, 0, 2)) # Resize to --> [batch, seq_len, ntoken]
        x = self.avg_pool(x)
        # x = self.self_attn_pool(x).unsqueeze(2)
        x = torch.permute(x, (0, 2, 1))
        #x = x[:, :, 0].unsqueeze(2)

        #x = torch.permute(x, (0, 2, 1))  # Resize to --> [batch, 1, seq_len]

        # Use 2-layer FC network to predict the output
        # x = self.dropout(F.relu(self.decoder(x)))
        # print(x.shape)
        x = self.decoder(x)
        # x *= self.norm_factor
        return x


class SelfAttentionPooling(nn.Module):
    """
    Implementation of SelfAttentionPooling
    Original Paper: Self-Attention Encoding and Pooling for Speaker Recognition
    https://arxiv.org/pdf/2008.01077v1.pdf
    """

    def __init__(self, input_dim):
        super(SelfAttentionPooling, self).__init__()
        self.W = nn.Linear(input_dim, 1)

    def forward(self, batch_rep):
        """
        input:
            batch_rep : size (N, T, H), N: batch size, T: sequence length, H: Hidden dimension

        attention_weight:
            att_w : size (N, T, 1)

        return:
            utter_rep: size (N, H)
        """
        softmax = nn.functional.softmax
        att_w = softmax(self.W(batch_rep).squeeze(-1)).unsqueeze(-1)
        utter_rep = torch.sum(batch_rep * att_w, dim=1)

        return utter_rep


class NoamOpt:
    "Optim wrapper that implements rate."

    def __init__(self, model_size, warmup, optimizer):
        self.optimizer = optimizer
        self._step = 0
        self.warmup = warmup
        self.model_size = model_size
        self._rate = 0

    def state_dict(self):
        """Returns the state of the warmup scheduler as a :class:`dict`.
        It contains an entry for every variable in self.__dict__ which
        is not the optimizer.
        """
        return {key: value for key, value in self.__dict__.items() if key != 'optimizer'}

    def load_state_dict(self, state_dict):
        """Loads the warmup scheduler's state.
        Arguments:
            state_dict (dict): warmup scheduler state. Should be an object returned
                from a call to :meth:`state_dict`.
        """
        self.__dict__.update(state_dict)

    def step(self):
        "Update parameters and rate"
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self._rate = rate
        self.optimizer.step()

    def rate(self, step=None):
        "Implement `lrate` above"
        if step is None:
            step = self._step
        return (self.model_size ** (-0.5) *
                min(step ** (-0.5), step * self.warmup ** (-1.5)))


class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        # x = x + self.pe[:x.size(0)]
        #pos = torch.tensor(np.linspace(0, 1, 256)).unsqueeze(1).unsqueeze(1)
        #x = x.permute
        x = x + self.pe[:x.size(0), :]

        #return self.dropout(x)
        return x


if __name__ == '__main__':
    d_model = 256
    seq_len = 300
    model = TransformerModel(ntoken=200, seq_len=seq_len, d_model=d_model, nhead=8, d_hid=2048, nlayers=8, norm_factor=1)
    model.train()
    model.to(torch.device('cuda:0'))

    loss_fn = nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    #optimizer = NoamOpt(d_model, 500,, lr=1e-9, betas=(0.9, 0.98), eps=1e-9))
    #                    torch.optim.Adam(model.parameters()

    x = torch.Tensor(np.loadtxt("C:/Users/ruben/Documents/thesis/data/vipl/split_traces/p1_v1_source1_0.npy"))
    x = x[:, :seq_len].unsqueeze(0)
    x = x.type(torch.FloatTensor).to(torch.device('cuda:0'))

    #x = 2 * torch.rand(200,1,1) - 1
    #x = x * 100 + 100
    #x = x.type(torch.IntTensor)
    #print(x)

    for i in range(1000):
        output = model(x)
        print(output.item())
        loss = loss_fn(output, torch.Tensor([0.3]).to(torch.device('cuda:0')).view(output.shape))
        loss.backward()
        optimizer.step()


