import numpy as np
import glob

import config


def extract_hr(path):
    files = glob.glob(path + '*.csv')
    for file in files:
        file = file.replace('\\', '/')
        data = np.loadtxt(file, delimiter=',', skiprows=1).astype(int)

        beats = np.bincount(data[:, 2])[1]
        minutes = (data[-1, 0] - data[0, 0]) / 1000 / 60
        hr = beats / minutes

        fname = file.split('/')[-1].split('.')[0]
        np.savetxt(config.PURE_FILTERED_TARGET_PATH + fname + '.csv', [hr], delimiter=",")



if __name__ == '__main__':
    path = config.PURE_TARGET_SIGNAL_DIR
    extract_hr(path)
