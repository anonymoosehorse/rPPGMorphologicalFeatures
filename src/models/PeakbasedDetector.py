import torch
from torch import nn
from torch.nn import functional as F

import numpy as np
from matplotlib import pyplot as plt

from ..utils.signal_processing import detect_peaks_torch,fir_bp_filter
from ..utils.pre_processing import get_wave_properties_torch

class PeakbasedDetector(nn.Module):
    def __init__(self, target_name, signal_fps):
        super().__init__()
        self.target_name = target_name
        self.signal_fps = signal_fps        
        self.dummy_conv = nn.Conv1d(1,1,1)
    
    @torch.no_grad()
    def forward(self, signal,times):                        
        ## Correct for the fact that the signal is in seconds 
        times = times * 1000.0
        res_list = []
        for sig,time in zip(signal,times):
            # sig_filt = fir_bp_filter(sig,self.signal_fps,order=51,cutoffs=[0.5,6])
            # sig_filt = torch.tensor(sig_filt.copy()).to(signal.device)

            peak_idcs,valley_idcs = detect_peaks_torch(sig,0.3)

            # peak_sig = np.zeros_like(sig)
            # peak_sig[peak_idcs] = 1
            # peak_sig[valley_idcs] = -1

            # plt.plot(time,peak_sig)

            res = get_wave_properties_torch(sig,time,peak_idcs,valley_idcs,self.signal_fps)
            
            if isinstance(self.target_name,list):
                res_dict = {key:torch.mean(torch.tensor(res[key])) for key in self.target_name}
                res_list.append(res_dict)
            else:
                res_list.append(torch.mean(torch.tensor(res[self.target_name])))
        
        if isinstance(self.target_name,list):
            output = {}
            for key in res_list[0].keys():
                output[key] = torch.tensor([res[key] for res in res_list]).to(signal.device)
        else:
            output = torch.tensor(res_list).to(signal.device)

        return output
