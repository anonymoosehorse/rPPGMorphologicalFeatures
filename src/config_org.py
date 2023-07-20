import torch
import sys

haarcascade_url = "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_alt2.xml"
eye_cascade_url = "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_eye.xml"

# HOME_DIR = "C:/Users/ruben/Documents/thesis/src/"
HOME_DIR = "/tudelft.net/staff-umbrella/StudentsCVlab/rsangers/data/"
CHECKPOINT_PATH = HOME_DIR + "../thesis/checkpoint/"
# CHECKPOINT_PATH = "/tudelft.net/staff-umbrella/StudentsCVlab/rsangers/thesis/plots/vicar/1d resnet1d gt/pwa/2022-06-13 20:25:20.876393/running_model_0.007498822174966335.pt"
# CHECKPOINT_PATH = "/tudelft.net/staff-umbrella/StudentsCVlab/rsangers/thesis/plots/vipl/1d resnet1d gt/pwa/2022-06-13 19:48:54.775641/running_model_0.014637022744864225.pt"
# CHECKPOINT_PATH = "/tudelft.net/staff-umbrella/StudentsCVlab/rsangers/thesis/plots/vipl/1d resnet1d gt/rt/2022-06-13 19:47:02.124145/running_model_20.38557783762614.pt"
# CHECKPOINT_PATH = "/tudelft.net/staff-umbrella/StudentsCVlab/rsangers/thesis/plots/vicar/1d resnet1d gt/rt/2022-06-13 20:24:48.262238/running_model_7.972048799196879.pt"
PLOT_PATH = HOME_DIR + "../thesis/plots/"
if len(sys.argv) == 8:
    DATASET = sys.argv[1]
    MODEL = sys.argv[2]
    DATA_DIM = sys.argv[3]
    TARGET = sys.argv[4]
    USE_GT = sys.argv[5] == "True"
    EPOCHS = int(sys.argv[6])
    FOLD_NR = int(sys.argv[7])
else:
    DATASET = "vicar"
    DATA_DIM = "1d"
    USE_GT = False
    TARGET = "hr"

USE_YUV = False
#VIPL_MAX = 0.366
#VIPL_MIN = -0.1431
if USE_GT:
    VIPL_MAX = 1.001
    VIPL_MIN = -0.001
    VICAR_MAX = 65414
    VICAR_MIN = 10816
else:
    VIPL_MAX = 0.0925
    VIPL_MIN = -0.3398
    VICAR_MAX = 0.04123
    VICAR_MIN = -0.03031
#VICAR_PWA_MEAN = 10194.39
#VICAR_AREA_MEAN = 13397290

CHECKPOINT_FILE = "/tudelft.net/staff-umbrella/StudentsCVlab/rsangers/thesis/src/model_checkpoints.csv"
RESULTS_FILE = "/tudelft.net/staff-umbrella/StudentsCVlab/rsangers/thesis/src/results.txt"

# Original data
PURE_TRACES_PATH = "/tudelft.net/staff-umbrella/StudentsCVlab/jjmkok/PixelTraces/"
VIPL_TRACES_PATH = "/tudelft.net/staff-umbrella/StudentsCVlab/jjmkok/VIPL_traces/"
VICAR_TRACES_PATH = "/tudelft.net/staff-bulk/ewi/insy/VisionLab/mbittner/VicarPPGBeyond/SkinpixelFiles/FaceReader9beta2Mask/"
FACE_DATA_DIR = HOME_DIR + "../VIPL-HR - original/data/"
TARGET_SIGNAL_DIR = HOME_DIR + DATASET + "/clean_hr/"

# Filtered traces
TRACES_FILTERED_PATH = HOME_DIR + DATASET + "/traces_filtered5/"

DELAY = 0

if USE_GT:
    VICAR_FOLDS = [[[6, 8], [1, 10]]]
    VIPL_FOLDS = [[[19, 62, 86, 67, 37, 77, 40, 7, 36, 100, 83, 89, 6, 45, 32, 22],
                   [33, 97, 98, 69, 71, 103, 9, 59, 43, 12, 104, 50, 88, 90, 27, 61]]]
else:
    VICAR_FOLDS = [[[3], [16]],
                   [[9], [7]],
                   [[2], [14]],
                   [[10], [15]],
                   [[8], [13]],
                   [[11], [1]],
                   [[4], [6]]]
    VIPL_FOLDS = [[[22, 29, 84, 93, 99, 94, 25], [67, 72, 20, 4, 48, 70, 69]],
                  [[66, 73, 51, 54, 15, 97, 83], [95, 55, 27, 58, 14, 37, 52]],
                  [[40, 38, 47, 91, 6, 7, 100], [45, 56, 63, 68, 86, 32, 10]],
                  [[75, 103, 78, 21, 92, 102, 39], [62, 81, 46, 1, 42, 36, 106]],
                  [[35, 96, 80, 49, 101, 64, 43], [26, 87, 82, 12, 33, 53, 23]],
                  [[65, 88, 85, 41, 107, 13, 59], [30, 104, 76, 105, 61, 31, 19]],
                  [[28, 5, 50, 3, 89, 77, 24], [17, 16, 11, 57, 2, 98, 79]]]
    #VICAR_FOLDS = [[[6, 8], [1, 10]]]
    #VIPL_FOLDS = [[[19, 62, 86, 67, 37, 77, 40, 7, 36, 100, 83, 89, 6, 45, 32, 22],
    #               [33, 97, 98, 69, 71, 103, 9, 59, 43, 12, 104, 50, 88, 90, 27, 61]]]
#

# Split signal and hr
# SPLIT_TRACES = HOME_DIR + DATASET + "/split_traces_" + str(DELAY) + "/"
SPLIT_TRACES = HOME_DIR + DATASET + "/split_traces5_1D/"

SPLIT_STMAPS = HOME_DIR + DATASET + "/split_stmaps4/"
SPLIT_STMAPS_OLD = HOME_DIR + DATASET + "/split_stmaps2/"
# SPLIT_STMAPS_FILTERED = HOME_DIR + DATASET + "/split_stmaps_filtered2/"
SPLIT_STMAPS_FILTERED = HOME_DIR + DATASET + "/split_stmaps_unfiltered/"
SPLIT_STMAPS_STRIDE = HOME_DIR + DATASET + "/split_stmaps2_stride/"

# RhythmNet training data
HR_DATA_PATH = HOME_DIR + DATASET + "/hr_rhythmnet/"
ST_MAPS_PATH = HOME_DIR + DATASET + "/IBIS_traces/"

FIVESECONDS_SIGNAL = HOME_DIR + DATASET + "/fiveseconds_signal/"
FIVESECONDS_HR = HOME_DIR + DATASET + "/fiveseconds_hr/"
FIVESECONDS_CWT = HOME_DIR + DATASET + "/fiveseconds_cwt/"

TENSECONDS_SIGNAL = HOME_DIR + DATASET + "/tenseconds_signal/"
TENSECONDS_HR = HOME_DIR + DATASET + "/tenseconds_hr/"
TENSECONDS_CWT = HOME_DIR + DATASET + "/tenseconds_cwt/"

# CWTNet training data
SPLIT_CWT = HOME_DIR + DATASET + "/split_cwt5/"
# SPLIT_CWT = HOME_DIR + DATASET + "/split_cwt_gt_augment/"

if DATA_DIM == '1d':
    DATA_PATH = SPLIT_TRACES
    if USE_GT:
        TARGET_PATH = DATA_PATH[:-1] + "_gt_" + TARGET + "/"
    else:
        TARGET_PATH = DATA_PATH[:-1]+"_"+TARGET+"/"
elif DATA_DIM == '2d':
    DATA_PATH = SPLIT_CWT
    if USE_GT:
        TARGET_PATH = SPLIT_TRACES[:-1] + "_gt_" + TARGET + "/"
    else:
        TARGET_PATH = SPLIT_TRACES[:-1]+"_"+TARGET+"/"
else:
    DATA_PATH = SPLIT_STMAPS_FILTERED
    # DATA_PATH = SPLIT_STMAPS
    if USE_GT:
        TARGET_PATH = SPLIT_STMAPS[:-1] + "_gt_" + TARGET + "/"
    else:
        TARGET_PATH = SPLIT_STMAPS[:-1]+"_"+TARGET+"/"

if USE_GT:
    DATA_PATH = DATA_PATH[:-1] + "_gt/"

TRAINSET = DATA_PATH + "trainset.npy"
TESTSET = DATA_PATH + "testset.npy"
VALSET = DATA_PATH + "valset.npy"

# Unsplit CWT data
CWT_FILTERED_PATH = HOME_DIR + DATASET + "/cwt_filtered/"
FILTERED_TARGET_PATH = HOME_DIR + DATASET + "/filtered_target/"
CWT_DATA_PATH = HOME_DIR + DATASET + "/cwt_data/"
CWT_TARGET_PATH = HOME_DIR + DATASET + "/cwt_target/"

# Ground truth split signal
SPLIT_SIGNAL_DIR = HOME_DIR + DATASET + "/split_signal/"

USE_CHECKPOINT = False
DEVICE = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

OPTIMIZER = 'adam'
BATCH_SIZE = 1
EPOCHS_VAL = 1
CLIP_SIZE = 300
lr = 1e-3
scale = 1
weight = 100
use_gru = True

IMAGE_WIDTH = 300
IMAGE_HEIGHT = 75
NUM_WORKERS = 0
GRU_TEMPORAL_WINDOW = 6