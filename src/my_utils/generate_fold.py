import glob
import random

import config
import numpy as np
import os

def generate_fold():
    test_nrs = [74, 100, 21, 10, 65, 90, 57, 58, 101, 59, 64, 7, 49, 43, 62, 25, 50, 105, 35, 81, 12]
    testset = []
    all_paths = glob.glob(config.ST_MAPS_PATH + "p*.npy")

    for nr in test_nrs:
        st_paths = glob.glob(config.ST_MAPS_PATH + "p"+str(nr)+"_v*.npy")
        for st_path in st_paths:
            testset.append(st_path)

    trainset = [path for path in all_paths if path not in testset]

    np.save(config.ST_MAPS_PATH+"trainset.npy", trainset)
    np.save(config.ST_MAPS_PATH+"testset.npy", testset)

    print(len(np.load(config.ST_MAPS_PATH+"trainset.npy")))
    print(len(np.load(config.ST_MAPS_PATH+"testset.npy")))

def generate_fold_cwt(path, dataset):
    if dataset == "vicar":
        # test_nrs = ["03", "04", "11"]
        test_nrs = ["06", "08"]
        val_nrs = ["01", "10"]
    else:
        test_nrs = [19, 62, 86, 67, 37, 77, 40, 7, 36, 100, 83, 89, 6, 45, 32, 22]
        val_nrs = [33, 97, 98, 69, 71, 103, 9, 59, 43, 12, 104, 50, 88, 90, 27, 61]
    trainset = []
    testset = []
    valset = []

    i_paths = glob.glob(path + "i*.csv")
    for p in i_paths:
        p = p.replace('\\', '/')
        fname = p.split('/')[-1][2:]  # remove i_ from name

        if dataset == "vicar":
            p_nr = fname.split('_')[0]
        else:
            p_nr = int(fname.split('_')[0][1:])
        if p_nr in test_nrs:
            testset.append(path + fname)
        elif p_nr in val_nrs:
            valset.append(path + fname)
        else:
            trainset.append(path + fname)

    np.save(path+"trainset.npy", trainset)
    np.save(path+"testset.npy", testset)
    np.save(path+"valset.npy", valset)

    print(len(np.load(path+"trainset.npy")))
    print(len(np.load(path+"testset.npy")))
    print(len(np.load(path + "valset.npy")))


def generate_fold_simulated(path):
    trainset = []
    testset = []

    i_paths = glob.glob(path + "i*.csv")
    all_nrs = list(range(len(i_paths)))
    test_nrs = random.sample(all_nrs, 200)
    print(test_nrs)

    for i, p in enumerate(i_paths):
        p = p.replace('\\', '/')
        fname = p.split('/')[-1][2:]  # remove i_ from name

        if i in test_nrs:
            testset.append(path + fname)
        else:
            trainset.append(path + fname)

    np.save(path+"trainset.npy", trainset)
    np.save(path+"testset.npy", testset)

    print(len(np.load(path+"trainset.npy")))
    print(len(np.load(path+"testset.npy")))


def generate_fold_test():
    trainset = glob.glob(config.ST_MAPS_PATH + "p*.npy")
    testset = []
    for i in trainset:
        testset.append(i.replace(os.sep, '/'))
    trainset = testset
    print(trainset)
    np.save(config.CWT_DATA_PATH+"trainset.npy", trainset)
    np.save(config.CWT_DATA_PATH+"testset.npy", testset)


def generate_fold_1D(path, dataset):
    if dataset == "vicar":
        # test_nrs = ["03", "04", "11"]
        test_nrs = ["06", "08"]
        val_nrs = ["01", "10"]
    else:
        test_nrs = [19, 62, 86, 67, 37, 77, 40, 7, 36, 100, 83, 89, 6, 45, 32, 22]
        val_nrs = [33, 97, 98, 69, 71, 103, 9, 59, 43, 12, 104, 50, 88, 90, 27, 61]
    trainset = []
    testset = []
    valset = []

    paths = glob.glob(path + "*.npy")
    for p in paths:
        p = p.replace('\\', '/')
        fname = p.split('/')[-1]
        if fname.endswith("set.npy"):
            continue

        if dataset == "vicar":
            p_nr = fname.split('_')[0]
        else:
            p_nr = int(fname.split('_')[0][1:])
        if p_nr in test_nrs:
            testset.append(path + fname)
        elif p_nr in val_nrs:
            valset.append(path + fname)
        else:
            trainset.append(path + fname)

    np.save(path+"trainset.npy", trainset)
    np.save(path+"testset.npy", testset)
    np.save(path+"valset.npy", valset)

    print(len(np.load(path+"trainset.npy")))
    print(len(np.load(path+"testset.npy")))
    print(len(np.load(path+"valset.npy")))


def generate_fold_stmaps(path, dataset):
    if dataset == "vicar":
        # test_nrs = ["01", "06", "10"]
        test_nrs = ["06", "08"]
        val_nrs = ["01", "10"]
    elif dataset == "vipl":
        # test_nrs = [74, 100, 21, 10, 65, 90, 57, 58, 101, 59, 64, 7, 49, 43, 62, 25, 50, 105, 35, 81, 12]
        test_nrs = [19, 62, 86, 67, 37, 77, 40, 7, 36, 100, 83, 89, 6, 45, 32, 22]
        val_nrs = [33, 97, 98, 69, 71, 103, 9, 59, 43, 12, 104, 50, 88, 90, 27, 61]
    else:
        test_nrs = ["01", "06", "10"]
    trainset = []
    testset = []
    valset = []

    paths = glob.glob(path + "*.npy")
    for p in paths:
        p = p.replace('\\', '/')
        fname = p.split('/')[-1]

        if fname.endswith("set.npy"):
            continue

        if dataset == "vicar":
            p_nr = fname.split('_')[0]
        elif dataset == "pure":
            p_nr = fname.split('-')[0]
        else:
            p_nr = int(fname.split('_')[0][1:])
        if p_nr in test_nrs:
            testset.append(path + fname)
        elif p_nr in val_nrs:
            valset.append(path+fname)
        else:
            trainset.append(path + fname)

    np.save(path+"trainset.npy", trainset)
    np.save(path+"testset.npy", testset)
    np.save(path+"valset.npy", valset)

    print(len(np.load(path+"trainset.npy")))
    print(len(np.load(path+"testset.npy")))
    print(len(np.load(path + "valset.npy")))



if __name__ == '__main__':
    dataset = "vipl"
    dataset = "vicar"
    # data_path = f"/tudelft.net/staff-umbrella/StudentsCVlab/rsangers/data/{dataset}/split_stmaps3/"
    data_path = f"D:\\Projects\\Waveform\\Code\\AlternativeRubenCode\\waveform_feature_estimation\\OutputFolder\\data\\{dataset}\\split_traces5_1D" + r"\\"

    # generate_fold_cwt(data_path, dataset)
    generate_fold_1D(data_path, dataset)
    # generate_fold_simulated(config.SPLIT_CWT)
    # generate_fold_stmaps(data_path, dataset)
