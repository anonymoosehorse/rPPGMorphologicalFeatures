from pathlib import Path

class DataPaths(object):

    def __init__(self,cfg):
        self.HOME_DIR = Path(cfg.dataset.root)
        DATASET = cfg.dataset.name
        self.DATASET = DATASET
        DATA_DIM = cfg.model.data_dim
        self.DATA_DIM = DATA_DIM
        USE_GT = cfg.dataset.use_gt
        self.USE_GT = USE_GT
        TARGET = cfg.model.target
        self.TARGET = TARGET

        self.CHECKPOINT_PATH = self.HOME_DIR.parent / Path("thesis/checkpoint/")
        self.PLOT_PATH = self.HOME_DIR.parent / Path("thesis/plots/")
        self.FACE_DATA_DIR = self.HOME_DIR.parent / "VIPL-HR - original/data"
        self.TARGET_SIGNAL_DIR = self.HOME_DIR / DATASET / "clean_hr"

        self.TRACES_FILTERED_PATH = self.HOME_DIR / DATASET / "traces_filtered5"
        self.SPLIT_TRACES = self.HOME_DIR / DATASET / "split_traces5_1D"

        self.SPLIT_STMAPS = self.HOME_DIR / DATASET / "split_stmaps4"
        self.SPLIT_STMAPS_OLD = self.HOME_DIR / DATASET / "split_stmaps2"
        # SPLIT_STMAPS_FILTERED = self.HOME_DIR / DATASET / "split_stmaps_filtered2"
        self.SPLIT_STMAPS_FILTERED = self.HOME_DIR / DATASET / "split_stmaps_unfiltered"
        self.SPLIT_STMAPS_STRIDE = self.HOME_DIR / DATASET / "split_stmaps2_stride"

        # RhythmNet training data
        self.HR_DATA_PATH = self.HOME_DIR / DATASET / "hr_rhythmnet"
        self.ST_MAPS_PATH = self.HOME_DIR / DATASET / "IBIS_traces"

        self.FIVESECONDS_SIGNAL = self.HOME_DIR / DATASET / "fiveseconds_signal"
        self.FIVESECONDS_HR = self.HOME_DIR / DATASET / "fiveseconds_hr"
        self.FIVESECONDS_CWT = self.HOME_DIR / DATASET / "fiveseconds_cwt"

        self.TENSECONDS_SIGNAL = self.HOME_DIR / DATASET / "tenseconds_signal"
        self.TENSECONDS_HR = self.HOME_DIR / DATASET / "tenseconds_hr"
        self.TENSECONDS_CWT = self.HOME_DIR / DATASET / "tenseconds_cwt"

        # CWTNet training data
        self.SPLIT_CWT = self.HOME_DIR / DATASET / "split_cwt5"
        # SPLIT_CWT = self.HOME_DIR + DATASET + "/split_cwt_gt_augment/"

        if DATA_DIM == '1d':
            self.DATA_PATH = self.SPLIT_TRACES
            if USE_GT:
                self.TARGET_PATH = self.DATA_PATH.parent / f"{self.DATA_PATH.stem}_gt_{TARGET}"
            else:
                self.TARGET_PATH = self.DATA_PATH.parent / f"{self.DATA_PATH.stem}_{TARGET}"
        elif DATA_DIM == '2d':
            self.DATA_PATH = self.SPLIT_CWT
            if USE_GT:
                self.TARGET_PATH = self.SPLIT_TRACES.parent / f"{self.SPLIT_TRACES.stem}_gt_{TARGET}"
            else:
                self.TARGET_PATH = self.SPLIT_TRACES.parent / f"{self.SPLIT_TRACES.stem}_{TARGET}"
        else:
            self.DATA_PATH = self.SPLIT_STMAPS_FILTERED
            # DATA_PATH = SPLIT_STMAPS
            if USE_GT:
                self.TARGET_PATH = self.SPLIT_STMAPS.parent / f"{self.SPLIT_STMAPS.stem}_gt_{TARGET}"        
            else:
                self.TARGET_PATH = self.SPLIT_STMAPS.parent / f"{self.SPLIT_STMAPS.stem}_{TARGET}"                

        if USE_GT:
            self.DATA_PATH = self.DATA_PATH.parent / f"{self.DATA_PATH.stem}_gt"

        self.TRAINSET = self.DATA_PATH / "trainset.npy"
        self.TESTSET = self.DATA_PATH / "testset.npy"
        self.VALSET = self.DATA_PATH / "valset.npy"

        # Unsplit CWT data
        self.CWT_FILTERED_PATH = self.HOME_DIR / DATASET / "cwt_filtered"
        self.FILTERED_TARGET_PATH = self.HOME_DIR / DATASET / "filtered_target"
        self.CWT_DATA_PATH = self.HOME_DIR / DATASET / "cwt_data"
        self.CWT_TARGET_PATH = self.HOME_DIR / DATASET / "cwt_target"

        # Ground truth split signal
        self.SPLIT_SIGNAL_DIR = self.HOME_DIR / DATASET / "split_signal"


class DataFolds(object):

    def __init__(self,cfg):
        if cfg.dataset.use_gt:
            self.VICAR_FOLDS = [[[6, 8], [1, 10]]]
            self.VIPL_FOLDS = [[[19, 62, 86, 67, 37, 77, 40, 7, 36, 100, 83, 89, 6, 45, 32, 22],
                        [33, 97, 98, 69, 71, 103, 9, 59, 43, 12, 104, 50, 88, 90, 27, 61]]]
        else:
            self.VICAR_FOLDS = [[[3], [16]],
                        [[9], [7]],
                        [[2], [14]],
                        [[10], [15]],
                        [[8], [13]],
                        [[11], [1]],
                        [[4], [6]]]
            
            self.VIPL_FOLDS = [[[22, 29, 84, 93, 99, 94, 25], [67, 72, 20, 4, 48, 70, 69]],
                        [[66, 73, 51, 54, 15, 97, 83], [95, 55, 27, 58, 14, 37, 52]],
                        [[40, 38, 47, 91, 6, 7, 100], [45, 56, 63, 68, 86, 32, 10]],
                        [[75, 103, 78, 21, 92, 102, 39], [62, 81, 46, 1, 42, 36, 106]],
                        [[35, 96, 80, 49, 101, 64, 43], [26, 87, 82, 12, 33, 53, 23]],
                        [[65, 88, 85, 41, 107, 13, 59], [30, 104, 76, 105, 61, 31, 19]],
                        [[28, 5, 50, 3, 89, 77, 24], [17, 16, 11, 57, 2, 98, 79]]]



class DataFoldsNew(object):
        
    VICAR_GT_FOLDS = [[[6, 8], [1, 10]]]    
    
    VIPL_GT_FOLDS = [[[19, 62, 86, 67, 37, 77, 40, 7, 36, 100, 83, 89, 6, 45, 32, 22],
                        [33, 97, 98, 69, 71, 103, 9, 59, 43, 12, 104, 50, 88, 90, 27, 61]]]    
    
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

    def __init__(self,use_gt,dataset):
        self.use_gt = use_gt
        self.dataset = dataset
        pass

    def get_fold(self,fold_nr):
        if self.dataset == 'vicar':
            if self.use_gt:
                return self.VICAR_GT_FOLDS[fold_nr]
            else:
                return self.VICAR_FOLDS[fold_nr]
        elif self.dataset == 'vipl':
            if self.use_gt:
                return self.VIPL_GT_FOLDS[fold_nr]
            else:
                return self.VIPL_FOLDS[fold_nr]
            

class DataFolds(object):

    def __init__(self,cfg):
        if cfg.dataset.use_gt:
            self.VICAR_FOLDS = [[[6, 8], [1, 10]]]
            self.VIPL_FOLDS = [[[19, 62, 86, 67, 37, 77, 40, 7, 36, 100, 83, 89, 6, 45, 32, 22],
                        [33, 97, 98, 69, 71, 103, 9, 59, 43, 12, 104, 50, 88, 90, 27, 61]]]
        else:
            self.VICAR_FOLDS = [[[3], [16]],
                        [[9], [7]],
                        [[2], [14]],
                        [[10], [15]],
                        [[8], [13]],
                        [[11], [1]],
                        [[4], [6]]]
            
            self.VIPL_FOLDS = [[[22, 29, 84, 93, 99, 94, 25], [67, 72, 20, 4, 48, 70, 69]],
                        [[66, 73, 51, 54, 15, 97, 83], [95, 55, 27, 58, 14, 37, 52]],
                        [[40, 38, 47, 91, 6, 7, 100], [45, 56, 63, 68, 86, 32, 10]],
                        [[75, 103, 78, 21, 92, 102, 39], [62, 81, 46, 1, 42, 36, 106]],
                        [[35, 96, 80, 49, 101, 64, 43], [26, 87, 82, 12, 33, 53, 23]],
                        [[65, 88, 85, 41, 107, 13, 59], [30, 104, 76, 105, 61, 31, 19]],
                        [[28, 5, 50, 3, 89, 77, 24], [17, 16, 11, 57, 2, 98, 79]]]



class DatasetStats(object):

    def __init__(self,cfg):
        if cfg.dataset.use_gt:
            self.VIPL_MAX = 1.001
            self.VIPL_MIN = -0.001
            self.VICAR_MAX = 65414
            self.VICAR_MIN = 10816
        else:
            self.VIPL_MAX = 0.0925
            self.VIPL_MIN = -0.3398
            self.VICAR_MAX = 0.04123
            self.VICAR_MIN = -0.03031

