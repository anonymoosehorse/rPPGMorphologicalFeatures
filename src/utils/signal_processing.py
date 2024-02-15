import json
import numpy as np
from scipy.signal import butter,filtfilt,firls
import matplotlib.pyplot as plt
import glob
from tqdm import tqdm
import pycwt as wavelet
import torch

def pos_img(mat,axis=0,time_dim=0):
    mean = np.nanmean(mat,axis=time_dim)
    tn_mat = mat / (mean - 1)

    proj_mat = np.array([[0,1,-1],[-2,1,1]])

    s_mat = tn_mat @ proj_mat.T

    std_mat = np.nanstd(s_mat,axis=axis)
    alpha = std_mat[:,0] / std_mat[:,1]

    z_mat = s_mat[:,:,0] + alpha * s_mat[:,:,1]

    return z_mat

def fir_bp_filter(sig,fs,order,cutoffs=[0.5,6]):
    bands = np.array([0,cutoffs[0]-0.2,cutoffs[0],cutoffs[1],cutoffs[1]+0.2,0.5*fs]) / (0.5*fs)
    firls_coeff = firls(order,bands,[0,0,1,1,0,0])
    return filtfilt(firls_coeff,1,sig)

def pos(r, g, b):
    tn_r = temp_normalize(r)
    tn_g = temp_normalize(g)
    tn_b = temp_normalize(b)

    s1 = tn_g - tn_b
    s2 = tn_g + tn_b - 2 * tn_r

    alpha = np.nanstd(s1) / np.nanstd(s2)

    z = s1 + alpha * s2

    return z


def temp_normalize(signal):
    #print(signal[0])
    #print(np.nan(signal))
    mean = np.nanmean(signal)
    #print(mean)
    return signal / (mean - 1)


def butter_lowpass_filter(data, fs, order,cutoffs=[0.5,6]):
    nyq = 0.5 * fs
    # Get the filter coefficients
    # b, a = butter(order, normal_cutoff, btype='low', analog=False)
    b, a = butter(order, (cutoffs[0] / nyq, cutoffs[1]/ nyq), btype='bandpass', analog=False)
    y = filtfilt(b, a, data)

    return y

def detect_peaks(sig, peak_delta):
    peaks = []
    valleys = []

    # Normalization by the mean
    norm_sig = sig - np.nanmean(sig)
    delta = peak_delta * np.nanmax(norm_sig)

    mxpos = 0
    mnpos = 0
    lookformax = True
    mx = np.NINF
    mn = np.Inf

    for (i, temp) in enumerate(norm_sig):
        if (temp > mx):
            mx = temp
            mxpos = i
        if (temp < mn):
            mn = temp
            mnpos = i

        if (lookformax):

            if (temp < (mx - delta)):
                # numPeaks++
                peaks.append(mxpos)
                mn = temp
                mnpos = i
                lookformax = False

        else:

            if (temp > (mn + delta)):
                # numValleys++
                valleys.append(mnpos)
                mx = temp
                mxpos = i
                lookformax = True

    # Remove the peak in clipped areas
    # for (int i = 0 i < (parameters.LeftClip - 1) i++) peakers[i] = false
    # for (int i = peakers.Length - (parameters.RightClip + 1) i < len(peakers) i++) peakers[i] = false

    return peaks, valleys
def detect_peaks_torch(sig, peak_delta):
    peaks = []
    valleys = []

    # Normalization by the mean
    norm_sig = sig - torch.mean(sig)
    delta = peak_delta * torch.max(norm_sig)

    mxpos = 0
    mnpos = 0
    lookformax = True
    mx = torch.tensor(np.NINF)
    mn = torch.tensor(np.Inf)

    for (i, temp) in enumerate(norm_sig):
        if (temp > mx):
            mx = temp
            mxpos = i
        if (temp < mn):
            mn = temp
            mnpos = i

        if (lookformax):

            if (temp < (mx - delta)):
                # numPeaks++
                peaks.append(mxpos)
                mn = temp
                mnpos = i
                lookformax = False

        else:

            if (temp > (mn + delta)):
                # numValleys++
                valleys.append(mnpos)
                mx = temp
                mxpos = i
                lookformax = True

    # Remove the peak in clipped areas
    # for (int i = 0 i < (parameters.LeftClip - 1) i++) peakers[i] = false
    # for (int i = peakers.Length - (parameters.RightClip + 1) i < len(peakers) i++) peakers[i] = false

    return torch.tensor(peaks).to(sig.device), torch.tensor(valleys).to(sig.device)

def signal_to_cwt(time_s, signal,output_size=256):
    
    

    # COMPUTE SCALES
    sc_min = -1
    sc_max = -1
    sc = np.arange(0.2, 1000.01, 0.01)
    MorletFourierFactor = 4 * np.pi / (6 + np.sqrt(2 + 6 ** 2))
    freqs = 1 / (sc * MorletFourierFactor)
    for freq in freqs:
        if freq < 0.6 and sc_max == -1:
            sc_max = sc[freqs == freq][0]
        elif freq < 8 and sc_min == -1:
            sc_min = sc[freqs == freq][0]
    sc = np.array([sc_min, sc_max])

    # RESAMPLE SIGNAL OVER 256 VALUES
    time_interp = np.linspace(time_s[0], time_s[-1], output_size)
    signal_interp = np.interp(time_interp, time_s, signal)

    # STANDARDIZE SIGNAL
    signal_interp = (signal_interp - np.mean(signal_interp)) / np.std(signal_interp)

    # COMPUTE CWT
    wavelet_type = 'morlet'
    dt = np.mean(np.diff(time_interp))
    # # scales = 1 / (sc * MorletFourierFactor * dt)
    # widths_a = np.linspace(sc[0], sc[1], math.ceil((sc[1]-sc[0])/0.00555))
    # freqs_a = 1/(wavelet.Morlet().flambda() * widths_a)
    # res_a = wavelet.cwt(signal_interp, dt, freqs=freqs_a, wavelet=wavelet_type)
    ds = round((sc[1]-sc[0])/output_size,5)
    
    widths = sc[0] + np.arange(0, np.ceil((sc[1]-sc[0])/ds)) * ds
    # cwA = scipy.signal.cwt(signal_interp, scipy.signal.morlet2, widths=widths)
    # cwA2 = pywt.cwt(data=signal_interp, scales=widths, wavelet='morl', method='fft')

    # Create Wavelet object
    # mother_wavelet = wavelet.Morlet(6)  # You can adjust the parameter
    # freqs = 1/(wavelet.Morlet().flambda() * widths)
    f_lambda = 1.047197551196598 #Constant factor found in matlab
    freqs = 1/(f_lambda * widths) 
    res = wavelet.cwt(signal_interp, dt, freqs=freqs, wavelet=wavelet_type)

    return res[0]