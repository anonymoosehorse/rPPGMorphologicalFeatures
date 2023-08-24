from scipy import signal
import numpy as np
import matplotlib.pyplot as plt
import math
import pycwt as wavelet

def signal_to_cwt(time, signal,output_size=256):
    
    

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
    time_interp = np.linspace(time[0], time[-1], output_size)
    signal_interp = np.interp(time_interp, time, signal)

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
    widths = sc[0] + np.arange(0, math.ceil((sc[1]-sc[0])/ds)) * ds
    # cwA = scipy.signal.cwt(signal_interp, scipy.signal.morlet2, widths=widths)
    # cwA2 = pywt.cwt(data=signal_interp, scales=widths, wavelet='morl', method='fft')

    # Create Wavelet object
    # mother_wavelet = wavelet.Morlet(6)  # You can adjust the parameter
    # freqs = 1/(wavelet.Morlet().flambda() * widths)
    f_lambda = 1.047197551196598 #Constant factor found in matlab
    freqs = 1/(f_lambda * widths) 
    res = wavelet.cwt(signal_interp, dt, freqs=freqs, wavelet=wavelet_type)

    return res[0]










if __name__ == '__main__':
    # data = np.loadtxt("C:/Users/ruben/Documents/thesis/data/hr_data/p1_v1_source1.csv", delimiter=',', skiprows=1)
    # t = data[:, 0]
    # sig = data[:, 1]
    sig = np.loadtxt("tmp_signal.txt")
    time = np.arange(300) * 1/30.0
    cwA = signal_to_cwt(time,sig)

    sc_min = -1
    sc_max = -1
    sc = np.arange(0.2, 1000, 0.01)
    MorletFourierFactor = 4 * np.pi / (6 + np.sqrt(2 + 6 ^ 2))
    freqs = 1 / (sc*MorletFourierFactor)
    for i in range(len(freqs)):
        if freqs[i] < 0.6 and sc_max == -1:
            sc_max = sc[i]
        elif freqs[i] < 8 and sc_min == -1:
            sc_min = sc[i]

    sc = [sc_min, sc_max]

    # RESAMPLE SIGNAL OVER 256 VALUES
    time_interp = np.linspace(time[0], time[-1], 256)
    signal_interp = np.interp(time_interp, time, sig)

    # STANDARDIZE SIGNAL
    signal_interp = (signal_interp - np.mean(signal_interp)) / np.std(signal_interp)

    # widths = np.arange(1, 31)
    print(math.ceil((sc[1]-sc[0])/0.00555))
    widths = np.linspace(sc[0], sc[1], math.ceil((sc[1]-sc[0])/0.00555))
    print(widths)
    cwtmatr = signal.cwt(sig, signal.morlet2, widths)
    print(cwtmatr.real)

    plt.imshow(cwtmatr.real, extent=[0, 200, 0, 4], cmap='PRGn', aspect='auto',
               vmax=abs(cwtmatr.real).max(), vmin=-abs(cwtmatr.real).max())
    plt.show()
    print()