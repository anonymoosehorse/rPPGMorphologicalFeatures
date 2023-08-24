from pathlib import Path
import config
from my_utils.extract_signal import extract_signal,extract_signal_stmap
from my_utils.split_signals import split_signal2
from my_utils.generate_fold import generate_fold

if __name__ == '__main__':
    traces_path = config.VICAR_TRACES_PATH
    save_path = config.TRACES_FILTERED_PATH
    dataset = 'vipl'
    # traces_path = "C:/Users/ruben/Documents/thesis/data/vipl/raw_signal/"
    fps = 30

    # extract_signal(traces_path, save_path, fps, dataset)

    # data_path = "C:/Users/ruben/Documents/thesis/data/vipl/split_stmaps2/"
    data_path = config.SPLIT_STMAPS
    save_path = config.SPLIT_STMAPS_FILTERED

    extract_signal_stmap(data_path, save_path, fps)

    #plot_signal(traces_path)

    use_gt = False
    use_stride = False

    signal_path = str(config.TRACES_FILTERED_PATH) + "\\"
    # signal_path = config.ST_MAPS_PATH
    # signal_save_path = config.SPLIT_TRACES[:-1]+"_gt"
    signal_save_path = str(config.SPLIT_TRACES) + "\\"
    # signal_save_path = config.SPLIT_STMAPS

    dataset = config.DATASET
    if dataset == 'vicar':
        # gt_path = "/tudelft.net/staff-umbrella/StudentsCVlab/rsangers/VicarPPGBeyond_SpO2Alignment/Signals/"
        # gt_path = "/tudelft.net/staff-bulk/ewi/insy/VisionLab/mbittner/VicarPPGBeyond/cleanedSignals/"
        gt_path = str(Path("D:\Projects\Waveform\Data\VicarPPGBeyond\cleanedSignals")) + "\\"
    else:
        gt_path = str(config.TARGET_SIGNAL_DIR) + "\\"

    # delay = 500
    # signal_path = "C:/Users/ruben/Documents/thesis/data/vipl/traces_filtered/"
    #gt_path = "C:/Users/ruben/Documents/thesis/data/pure/clean_hr/"
    # signal_path = "C:/Users/ruben/Documents/thesis/data/vicar/vicar_traces_filtered/"
    # gt_path = "C:/Users/ruben/Documents/thesis/data/vicar/vicar_gt/"
    # peaks_path = "C:/Users/ruben/Documents/thesis/data/vipl_signal/"
    # signal_save_path = "C:/Users/ruben/Documents/thesis/data/test_cwt/"
    # hr_save_path = "C:/Users/ruben/Documents/thesis/data/fiveseconds_hr/"

    # correct_bpm(hr_save_path)
    # plot_bpms(hr_save_path)
    # split_stmap(signal_path, gt_path, signal_save_path, dataset, use_stride=use_stride)
    # split_signal(signal_path, 10, signal_save_path, signal_save_path)
    split_signal2(signal_path, gt_path, signal_save_path, dataset, use_gt=use_gt, use_stride=use_stride)
    # split_signal_time(signal_path, peaks_path, signal_save_path, hr_save_path, t)
    # split_signal_noisy(signal_path, peaks_path, signal_save_path, hr_save_path, signal_length, delay)
    # show_peaks(signal_path, peaks_path)


    dataset = "vipl"
    data_path = f"/tudelft.net/staff-umbrella/StudentsCVlab/rsangers/data/{dataset}/split_stmaps3/"

    # generate_fold_cwt(data_path, dataset)
    # generate_fold_1D(data_path, dataset)
    # generate_fold_simulated(config.SPLIT_CWT)
    generate_fold_stmaps(data_path, dataset)
