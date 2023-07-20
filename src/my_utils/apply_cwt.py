from scipy import signal
import numpy as np
import matplotlib.pyplot as plt
import math

if __name__ == '__main__':
    data = np.loadtxt("C:/Users/ruben/Documents/thesis/data/hr_data/p1_v1_source1.csv", delimiter=',', skiprows=1)
    t = data[:, 0]
    sig = data[:, 1]



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

    # widths = np.arange(1, 31)
    print(math.ceil((sc[1]-sc[0])/0.00555))
    widths = np.linspace(sc[0], sc[1], math.ceil((sc[1]-sc[0])/0.00555))
    print(widths)
    cwtmatr = signal.cwt(sig, signal.morlet2, widths)
    print(cwtmatr.real)

    plt.imshow(cwtmatr.real, extent=[0, 200, 0, 4], cmap='PRGn', aspect='auto',
               vmax=abs(cwtmatr.real).max(), vmin=-abs(cwtmatr.real).max())
    plt.show()