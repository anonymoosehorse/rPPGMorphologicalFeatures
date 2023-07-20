import glob
import math

import numpy
import numpy as np
import matplotlib.pyplot as plt
import statistics
from tqdm import tqdm
import config
import os
from PIL import Image
import cv2
import seaborn as sns

from my_utils.split_signals import detect_peaks
#from main2 import rmse, mae


def show_statistics(dataset, hr_path, save_name):
    if dataset == "vicar":
        test_nrs = ["06", "08"]
        val_nrs = ["01", "10"]
    else:
        test_nrs = [19, 62, 86, 67, 37, 77, 40, 7, 36, 100, 83, 89, 6, 45, 32, 22]
        val_nrs = [33, 97, 98, 69, 71, 103, 9, 59, 43, 12, 104, 50, 88, 90, 27, 61]

    files = glob.glob(hr_path + "*.npy")
    hrs = []
    for file in files:
        file = file.replace('\\', '/')
        pnr = file.split('/')[-1].split('_')[0]
        if dataset == 'vipl':
            pnr = int(pnr[1:])
        if pnr not in test_nrs and pnr not in val_nrs:
            hrs.append(np.loadtxt(file).item())
    print(f"Mean HR: {statistics.mean(hrs)}")
    print(f"Std. HR: {statistics.stdev(hrs)}")
    print(f"Size trainset: {len(hrs)}")

    hrs = np.asarray(hrs)
    q25, q75 = np.percentile(hrs, [25, 75])
    bin_width = 2 * (q75 - q25) * len(hrs) ** (-1 / 3)
    bins = round((hrs.max() - hrs.min()) / bin_width)
    plt.hist(hrs, bins=bins)
    plt.savefig(save_name)


def calculate_error(values, dataset, target_path):
    if dataset == "vicar":
        val_nrs = ["01", "10"]
    else:
        val_nrs = [33, 97, 98, 69, 71, 103, 9, 59, 43, 12, 104, 50, 88, 90, 27, 61]

    files = glob.glob(target_path[:-1]+"_hr/" + "*.npy")
    print(f"Dataset: {dataset}, {len(files)} files")
    hrs = []
    rts = []
    pwas = []
    areas = []
    for file in tqdm(files):
        file = file.replace('\\', '/')
        fname = file.split('/')[-1]
        pnr = file.split('/')[-1].split('_')[0]
        if dataset == 'vipl':
            pnr = int(pnr[1:])
        if pnr in val_nrs:
            if target_path.endswith("split_stmaps2/"):
                hrs.append(np.load(target_path[:-1]+"_hr/"+fname).item())
                rts.append(np.load(target_path[:-1] + "_rt/"+fname).item())
                pwas.append(np.load(target_path[:-1] + "_pwa/"+fname).item())
                areas.append(np.load(target_path[:-1] + "_area/"+fname).item())
            else:
                hrs.append(np.loadtxt(target_path[:-1] + "_hr/" + fname).item())
                rts.append(np.loadtxt(target_path[:-1] + "_rt/" + fname).item())
                pwas.append(np.loadtxt(target_path[:-1] + "_pwa/" + fname).item())
                areas.append(np.loadtxt(target_path[:-1] + "_area/" + fname).item())

    values_lists = []
    for v in values:
        values_lists.append([v for i in hrs])

    #for i in range(len(hrs)):
    #    hr_mae.append(abs(values[0] - hrs[i]))
    #    rt_mae.append(abs(values[1] - rts[i]))
    #    pwa_mae.append(abs(values[2] - pwas[i]))
    #    aup_mae.append(abs(values[3] - areas[i]))
    print(f"HR error: {mae(values_lists[0], hrs)}, {rmse(np.asarray(values_lists[0]), np.asarray(hrs))}")
    print(f"RT error: {mae(values_lists[1], rts)}, {rmse(np.asarray(values_lists[1]), np.asarray(rts))}")
    print(f"PWA error: {mae(values_lists[2], pwas)}, {rmse(np.asarray(values_lists[2]), np.asarray(pwas))}")
    print(f"AUP error: {mae(values_lists[3], areas)}, {rmse(np.asarray(values_lists[3]), np.asarray(areas))}")



def show_statistics_all(dataset, target_path, save_path):
    #if dataset == "vicar":
    #    test_nrs = ["06", "08"]
    #    val_nrs = ["01", "10"]
    #else:
    #    test_nrs = [19, 62, 86, 67, 37, 77, 40, 7, 36, 100, 83, 89, 6, 45, 32, 22]
    #    val_nrs = [33, 97, 98, 69, 71, 103, 9, 59, 43, 12, 104, 50, 88, 90, 27, 61]
    sns.set_theme()
    fig, axes = plt.subplots(4, 1, figsize=(16, 4))
    x_titles = ["HR", "RT", "PWA", "AUP"]

    files = glob.glob(target_path + "*_*.npy")
    hrs = []
    rts = []
    pwas = []
    areas = []
    for file in tqdm(files):
        file = file.replace('\\', '/')
        fname = file.split('/')[-1]
        #pnr = file.split('/')[-1].split('_')[0]
        #if dataset == 'vipl':
        #    pnr = int(pnr[1:])
        #if (set == 'trainset' and pnr not in test_nrs and pnr not in val_nrs) or (set == 'valset' and pnr in val_nrs):
        if target_path.endswith("split_stmaps_filtered/"):
            hrs.append(np.load(target_path[:-1] + "_hr/" + fname).item())
            rts.append(np.load(target_path[:-1] + "_rt/" + fname).item())
            pwas.append(np.load(target_path[:-1] + "_pwa/" + fname).item())
            areas.append(np.load(target_path[:-1] + "_area/" + fname).item())
        else:
            hrs.append(np.loadtxt(target_path[:-1] + "_hr/" + fname).item())
            rts.append(np.loadtxt(target_path[:-1] + "_rt/" + fname).item())
            pwas.append(np.loadtxt(target_path[:-1] + "_pwa/" + fname).item())
            areas.append(np.loadtxt(target_path[:-1] + "_area/" + fname).item())
    print(f"Size: {len(hrs)}")
    np.savetxt(save_path+dataset+"_statistics.csv", [hrs, rts, pwas, areas], delimiter=',')

    #for x, data in enumerate([hrs, rts, pwas, areas]):
    #    #print(f"Mean {x_titles[x]}: {statistics.mean(data)}")
    #
    #    # plt.figure()
    #    data_array = np.asarray(data)
    #    q25, q75 = np.percentile(data_array, [25, 75])
    #    bin_width = 2 * (q75 - q25) * len(data_array) ** (-1 / 3)
    #    bins = round((data_array.max() - data_array.min()) / bin_width)
    #    weights = np.ones_like(data_array) / len(data_array)
    #    axes[x].hist(data_array, bins=bins, weights=weights)
    #
    #    xlim = axes[x].get_xlim()
    #    ylim = axes[x].get_ylim()
    #
    #    text_positions = (xlim[0] + 0.05 * (xlim[1] - xlim[0]), ylim[1] - 0.15 * (ylim[1] - ylim[0]))
    #    axes[x].annotate(f"Mean: {statistics.mean(data)}", xy=text_positions)
    #    axes[x].set_title(x_titles[x])
    #
    #    #plt.xlabel(datatypes[idx])
    #    #plt.ylabel(f"Fraction of {set}")
    #    #plt.savefig(f"{save_name[:-4]}_{datatypes[idx]}.png")
    #plt.savefig(f"{save_path}{dataset}_histograms.png")




def generate_signal(signal_save_path, rt_save_path, speedup):
    dt = 0.01
    x = np.arange(0, 5, dt)
    y = - np.cos(x*2*math.pi)

    # peaks, valleys = detect_peaks(y, 0.3)
    # valleys.insert(0, 0)

    peaks = [50, 150, 250, 350, 450]
    valleys = [0, 100, 200, 300, 400]

    # peak_times = [x[p] for p in peaks]
    # peak_heights = [y[p] for p in peaks]
    # valley_times = [x[v] for v in valleys]
    # valley_heights = [y[v] for v in valleys]

    idx = 0
    for i in range(len(x[:-1])):
        i += 1
        if idx >= len(valleys):
            x[i] = x[i-1] + dt
        elif i > valleys[idx] and i < peaks[idx]:
            x[i] = x[i-1] + dt / speedup
        elif i == peaks[idx]:
            idx += 1
            x[i] = x[i - 1] + dt
        else:
            x[i] = x[i-1] + dt

    plt.plot(x, y)
    #plt.plot(peak_times, peak_heights, 'ro')
    #plt.plot(valley_times, valley_heights, 'go')
    plt.show()

    x = [i * 1000 for i in x]
    data = np.stack((x, y))
    rt = [500 / speedup]
    #np.savetxt(f"{signal_save_path}sim_signal_{speedup}.csv", data, delimiter=',')
    #np.savetxt(f"{rt_save_path}sim_signal_{speedup}.csv", rt, delimiter=',')


def combine_files(path, files, save_path):
    #files = glob.glob(path+"traces_*.seg")
    traces_files = []
    for f in files:
        if f.startswith("traces"):
            traces_files.append(f)
    files = traces_files

    file_name = path.split('/')[-2].split('.')[0]
    if file_name.startswith('video'):
        return
    data = np.expand_dims(np.loadtxt(path+files[0]), axis=0)
    files.pop(0)
    #datafiles = []
    for file in files:
        new_data = np.expand_dims(np.loadtxt(path+file), axis=0)
        data = np.concatenate((data, new_data), axis=0)
        #datafiles.append(new_data)

    # data = np.concatenate(datafiles, axis=0)
    np.save(save_path+file_name+".npy", data)
    #data = np.load(path+"combined_data.npy")
    print(data.shape)


def check_rois(path):
    files = glob.glob(f"{path}parent_*.seg")
    print(len(files))
    for file in files:
        data = np.loadtxt(file)
        for i in range(len(data) - 1):
            if data[i] != data[i+1] - 1:
                print(data[i])
                print(data[i+1])
                print(file)
                return
    print("All files correct")


def rename_vipl_files():
    path = "/tudelft.net/staff-umbrella/StudentsCVlab/rsangers/VIPL-HR - original/data/"
    video_files = glob.glob(path + '**/used_video.avi', recursive=True)
    print(f"Found {len(video_files)} in total")
    for video in video_files:
        video_path_split = video.split('/')
        video_path = "/".join(video_path_split[:-1])
        p = video_path_split[-4][1:]
        v = video_path_split[-3][-1]
        s = video_path_split[-2][-1]
        os.rename(video, os.path.join(video_path, f"p{p}_v{v}_s{s}.avi"))
    print("Successfully renamed "+str(len(video_files))+" files")


def combine_files_batch(data_path, save_path):
    for root, dirs, files in os.walk(data_path):
        if len(files) > 0 and files[0].endswith(".seg"):
            combine_files(root+'/', files, save_path)


def visualize_roi(data, roi_number):
    print(data.shape)
    r = data[:, roi_number, 0]
    g = data[:, roi_number, 1]
    b = data[:, roi_number, 2]
    t = np.arange(0, len(r), 1)
    plt.plot(t, r)
    plt.plot(t, g)
    plt.plot(t, b)
    plt.show()


def visualize_stmap(datapath):
    sns.set_theme(style='whitegrid')
    sns.set_context(context='paper')

    data = np.float32(np.load(datapath))
    data = (data - np.min(data)) / (np.max(data) - np.min(data)) * 255
    data = np.swapaxes(data, 0, 1)

    for i in range(len(data)):
        data[i] = (data[i] - np.min(data[i])) / (np.max(data[i]) - np.min(data[i])) * 255

    data = cv2.cvtColor(data, cv2.COLOR_RGB2YUV)
    data = (data - np.min(data)) / (np.max(data) - np.min(data))
    #data = cv2.resize(data, dsize=(2400, 2500), interpolation=cv2.INTER_NEAREST)
    print(np.max(data))
    plt.imshow(data)
    plt.xlabel('Time (s)')
    plt.ylabel('Superpixel')
    x_positions = np.arange(0, 251, 50)
    xlabels = [0, 2, 4, 6, 8, 10]
    plt.xticks(x_positions, xlabels)
    plt.show()

    #cv2.imshow("Image", data)
    #cv2.waitKey(0)
    #cv2.imwrite("stmap_example.png", data)


def calculate_error_LOOCV(data_path, dataset):
    features = ["hr", "rt", "pwa", "area"]
    hrs_per_pnr = []
    rts_per_pnr = []
    pwas_per_pnr = []
    areas_per_pnr = []

    if dataset == 'vipl':
        folds = config.VIPL_FOLDS
        participants = list(np.arange(1, 108))
        participant_index = dict(zip(participants, np.arange(len(participants))))
        for i, p in tqdm(enumerate(participants)):
            hrs_per_pnr.append([])
            rts_per_pnr.append([])
            pwas_per_pnr.append([])
            areas_per_pnr.append([])
            files = glob.glob(data_path + f"p{p}_*.npy")
            for file in files:
                fname = file.split("/")[-1]
                hrs_per_pnr[i].append(np.loadtxt(f"{data_path[:-1]}_hr/{fname}").item())
                rts_per_pnr[i].append(np.loadtxt(f"{data_path[:-1]}_rt/{fname}").item())
                pwas_per_pnr[i].append(np.loadtxt(f"{data_path[:-1]}_pwa/{fname}").item())
                areas_per_pnr[i].append(np.loadtxt(f"{data_path[:-1]}_area/{fname}").item())
    else:
        folds = config.VICAR_FOLDS
        participants = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16]
        participant_index = dict(zip(participants, np.arange(len(participants))))
        all_files = glob.glob(data_path + "*.npy")
        for i, p in tqdm(enumerate(participants)):
            hrs_per_pnr.append([])
            rts_per_pnr.append([])
            pwas_per_pnr.append([])
            areas_per_pnr.append([])
            files = []
            for file in all_files:
                if int(file.split("/")[-1].split("_")[0]) == p:
                    files.append(file)
            for file in files:
                fname = file.split("/")[-1]
                hrs_per_pnr[i].append(np.loadtxt(f"{data_path[:-1]}_hr/{fname}").item())
                rts_per_pnr[i].append(np.loadtxt(f"{data_path[:-1]}_rt/{fname}").item())
                pwas_per_pnr[i].append(np.loadtxt(f"{data_path[:-1]}_pwa/{fname}").item())
                areas_per_pnr[i].append(np.loadtxt(f"{data_path[:-1]}_area/{fname}").item())

    print("Loaded all data")

    hr_maes = []
    hr_rmses = []
    rt_maes = []
    rt_rmses = []
    pwa_maes = []
    pwa_rmses = []
    area_maes = []
    area_rmses = []
    for fold in tqdm(folds):
        trainset = [p for p in participants if p not in fold[0] and p not in fold[1]]
        train_hrs = []
        train_rts = []
        train_pwas = []
        train_areas = []
        val_hrs = []
        val_rts = []
        val_pwas = []
        val_areas = []
        test_hrs = []
        test_rts = []
        test_pwas = []
        test_areas = []
        for p in trainset:
            for i in range(len(hrs_per_pnr[participant_index[p]])):
                train_hrs.append(hrs_per_pnr[participant_index[p]][i])
                train_rts.append(rts_per_pnr[participant_index[p]][i])
                train_pwas.append(pwas_per_pnr[participant_index[p]][i])
                train_areas.append(areas_per_pnr[participant_index[p]][i])
        for p in fold[1]:
            for i in range(len(hrs_per_pnr[participant_index[p]])):
                val_hrs.append(hrs_per_pnr[participant_index[p]][i])
                val_rts.append(rts_per_pnr[participant_index[p]][i])
                val_pwas.append(pwas_per_pnr[participant_index[p]][i])
                val_areas.append(areas_per_pnr[participant_index[p]][i])
        for p in fold[0]:
            for i in range(len(hrs_per_pnr[participant_index[p]])):
                test_hrs.append(hrs_per_pnr[participant_index[p]][i])
                test_rts.append(rts_per_pnr[participant_index[p]][i])
                test_pwas.append(pwas_per_pnr[participant_index[p]][i])
                test_areas.append(areas_per_pnr[participant_index[p]][i])
        pred_hr = sum(train_hrs)/len(train_hrs)
        pred_rt = sum(train_rts) / len(train_rts)
        pred_pwa = sum(train_pwas) / len(train_pwas)
        pred_area = sum(train_areas) / len(train_areas)
        #print("Trainset average:")
        #print(pred_hr)
        #print(pred_rt)
        #print(pred_pwa)
        #print(pred_area)
        #print("Testset average:")
        #print(sum(test_hrs) / len(test_hrs))
        #print(sum(test_rts) / len(test_rts))
        #print(sum(test_pwas) / len(test_pwas))
        #print(sum(test_areas) / len(test_areas))
        #print("Validationset average:")
        print(sum(val_hrs)/len(val_hrs))
        print(sum(val_rts) / len(val_rts))
        print(sum(val_pwas) / len(val_pwas))
        print(sum(val_areas) / len(val_areas))
        print("---------------")

        hr_maes.append(mae(np.array([pred_hr for i in range(len(val_hrs))]), np.array(val_hrs)))
        hr_rmses.append(rmse(np.array([pred_hr for i in range(len(val_hrs))]), np.array(val_hrs)))
        rt_maes.append(mae(np.array([pred_rt for i in range(len(val_rts))]), np.array(val_rts)))
        rt_rmses.append(rmse(np.array([pred_rt for i in range(len(val_rts))]), np.array(val_rts)))
        pwa_maes.append(mae(np.array([pred_pwa for i in range(len(val_pwas))]), np.array(val_pwas)))
        pwa_rmses.append(rmse(np.array([pred_pwa for i in range(len(val_pwas))]), np.array(val_pwas)))
        area_maes.append(mae(np.array([pred_area for i in range(len(val_areas))]), np.array(val_areas)))
        area_rmses.append(rmse(np.array([pred_area for i in range(len(val_areas))]), np.array(val_areas)))
    print(hr_maes)
    print(hr_rmses)
    print(rt_maes)
    print(rt_rmses)
    print(pwa_maes)
    print(pwa_rmses)
    print(area_maes)
    print(area_rmses)


if __name__ == '__main__':
    dataset = 'vicar'
    data_path = f"/tudelft.net/staff-umbrella/StudentsCVlab/rsangers/data/{dataset}/split_traces5_1D_gt/"
    save_path = "/tudelft.net/staff-umbrella/StudentsCVlab/rsangers/thesis/src/"
    #calculate_error_LOOCV(data_path, dataset)
    show_statistics_all(dataset, data_path, save_path)

    #save_path = "/tudelft.net/staff-umbrella/StudentsCVlab/rsangers/data/pure/IBIS_traces/"
    #data_path = "/tudelft.net/staff-umbrella/StudentsCVlab/rsangers/thesis/src/results/tudelft.net/staff-umbrella/"
    #combine_files_batch(data_path, save_path)
    #files = glob.glob(path + "*.npy")
    #min_rois = np.Inf
    #max_rois = -np.Inf
    #for file in tqdm(files):
    #    data = np.load(file)
    #    min_rois = min(min_rois, data.shape[1])
    #    max_rois = min(max_rois, data.shape[1])

    #print(f"Minimum: {min_rois}, maximum: {max_rois}")

    # set = 'trainset'
    #dataset = "vicar"
    #target_path = f"/tudelft.net/staff-umbrella/StudentsCVlab/rsangers/data/{dataset}/split_traces4_1D/"
    #save_path = f"/tudelft.net/staff-umbrella/StudentsCVlab/rsangers/thesis/plots/"
    #show_statistics_all(dataset, target_path, save_path)

    #values_vipl_1d = [79.333, 204.07, 0.62374, 1.6146]
    #values_vipl_2d = [79.553, 202.95, 0.61952, 1.7942]
    #values_vicar_1d = [70.138, 188.74, 0.24763, 0.30423]
    #values_vicar_2d = [71.586, 185.59, 0.24736, 0.31485]
    #calculate_error(values_vipl_1d, dataset, target_path)

    # path = "C:/Users/ruben/Documents/thesis/data/vipl/split_stmaps_filtered/p1_v1_s1_0.npy"
    # path = "C:/Users/ruben/Documents/thesis/data/vipl/split_cwt/r_p1_v1_source1_0.csv"
    #path = "C:/Users/ruben/Documents/thesis/data/vipl/split_traces/p1_v1_source1_0.npy"
    # sns.set_theme(style='whitegrid')
    #sns.set_context(context='paper')

    #data = np.loadtxt(path, delimiter=',')
    #for i in range(len(data)):
    #    data[i] = 255 * (data[i] - np.min(data[i])) / (np.max(data[i]) - np.min(data[i]) + 0.000001)
    #plt.imshow(data)
    #plt.xlabel('Time (s)')
    #plt.ylabel('Superpixel')
    #x_positions = np.arange(0, 251, 50)
    #xlabels = [0, 2, 4, 6, 8, 10]
    #plt.xticks(x_positions, xlabels)
    #plt.savefig("C:/Users/ruben/Documents/Master CS/Master thesis/stmap.png")
    #plt.show()

    #data = np.loadtxt(path, delimiter=',')
    #plt.imshow(data, cmap='Blues')
    #plt.colorbar()
    #plt.xlabel('Time (s)')
    #plt.ylabel('Frequency (Hz)')
    #x_positions = np.arange(0, 256, 51)
    #xlabels = [0, 2, 4, 6, 8, 10]
    #plt.xticks(x_positions, xlabels)
    #y_positions = np.arange(21, 256, 21)
    #ylabels = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.3, 1.5, 1.8, 2.3, 3.1]
    #ylabels.reverse()
    #plt.yticks(y_positions, ylabels)
    #plt.show()

    #data = np.loadtxt(path)
    #t = [t/1000 for t in data[0] if t <= 10000]
    #y = data[1, 0:len(t)]
    #sns.lineplot(x=t, y=y)
    #plt.xlabel('Time (s)')
    #plt.ylabel('Amplitude')
    #plt.show()


    #path = "C:/Users/ruben/Documents/thesis/data/test_IBIS/p1_v5_s1.npy"
    #data = np.load("p1_v1_s1_0.npy")
    #print(data)
    #path = f"/tudelft.net/staff-umbrella/StudentsCVlab/rsangers/thesis/src/IBIS_temporal/build/results/videos/used_video.avi/"
    #save_path = "/tudelft.net/staff-umbrella/StudentsCVlab/rsangers/data/vipl/IBIS_traces/"
    #data_path = "/tudelft.net/staff-umbrella/StudentsCVlab/rsangers/thesis/src/results/tudelft.net/staff-umbrella/StudentsCVlab/rsangers/VIPL-HR-original/data/"
    #data = np.load("C:/Users/ruben/Documents/thesis/data/test_IBIS/p80_v1_s1.npy")

    #dataset = "vipl"
    #hr_path = f"/tudelft.net/staff-umbrella/StudentsCVlab/rsangers/data/{dataset}/split_traces2_hr_1D/"
    #save_name = f"/tudelft.net/staff-umbrella/StudentsCVlab/rsangers/thesis/plots/{dataset}2_hr.png"
    #signal_save_path = "C:/Users/ruben/Documents/thesis/data/simulated/"
    #rt_save_path = "C:/Users/ruben/Documents/thesis/data/simulated/"
    #signal_save_path = "/tudelft.net/staff-umbrella/StudentsCVlab/rsangers/data/simulated/signals/"
    #rt_save_path = "/tudelft.net/staff-umbrella/StudentsCVlab/rsangers/data/simulated/rts/"

    #for j in tqdm(range(1000)):
    #    speedup = np.random.uniform(0.5, 2)
    #    generate_signal(signal_save_path, rt_save_path, speedup)
    # generate_signal(signal_save_path, rt_save_path, 2)

    #show_statistics(dataset, hr_path, save_name)

    #files = glob.glob(hr_path + "*.csv")
    #hrs = []
    #for file in files:
    #    hrs.append(np.loadtxt(file).item())
    #print(statistics.mean(hrs))
    #print(statistics.stdev(hrs))

    #hrs = np.asarray(hrs)
    #q25, q75 = np.percentile(hrs, [25, 75])
    #bin_width = 2 * (q75 - q25) * len(hrs) ** (-1 / 3)
    #bins = round((hrs.max() - hrs.min()) / bin_width)
    #plt.hist(hrs, bins=bins, density=True)
    #plt.savefig(save_name)






