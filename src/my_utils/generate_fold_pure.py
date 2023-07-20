import numpy as np
import glob
import config


def generate_fold_cwt(path):

    test_nrs = ['01', '07']
    trainset = []
    testset = []

    i_paths = glob.glob(path + "i*.csv")

    for p in i_paths:
        p = p.replace('\\', '/')
        fname = p.split('/')[-1][2:]  # remove i_ from name

        p_nr = fname.split('-')[0]
        if p_nr in test_nrs:
            testset.append(path + fname)
        else:
            trainset.append(path + fname)

    np.save(path+"trainset.npy", trainset)
    np.save(path+"testset.npy", testset)

    print(len(np.load(path+"trainset.npy")))
    print(len(np.load(path+"testset.npy")))


if __name__ == '__main__':
    generate_fold_cwt(config.PURE_SPLIT_CWT)