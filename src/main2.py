import os
from comet_ml import Experiment
import torch
import numpy as np
import torch.nn as nn
import train_eval_functions
import config
import sys
from datetime import datetime
import glob
from pathlib import Path
import re

# from torch.utils.tensorboard import SummaryWriter
from utils.dataset import DataLoaderSTMaps, DataLoaderCWTNet, DataLoader1D
from utils.plot_scripts import plot_train_curve, bland_altman_plot, gt_vs_est
from utils.model_utils import load_model_if_checkpointed, save_model_checkpoint

from myModels.CWTNet import CWTNet
from myModels.CWTNet2 import CWTNet2
from myModels.TransformerModel import TransformerModel, NoamOpt
from myModels.CNN1D import CNN1D
from myModels.resnet1d import ResNet1D
from myModels.resnet2d import Resnet2D

# Needed in VIPL DATASET where each data item has a different number of frames/maps
def collate_fn(batch):
    batched_st_map, batched_targets = [], []
    # for data in batch:
    #     batched_st_map.append(data["st_maps"])
    #     batched_targets.append(data["target"])
    # # torch.stack(batched_output_per_clip, dim=0).transpose_(0, 1)
    return batch


def rmse(l1, l2):

    return np.sqrt(np.mean((l1-l2)**2))


def mae(l1, l2):

    return np.mean([abs(item1-item2)for item1, item2 in zip(l1, l2)])


def compute_criteria(target_hr_list, predicted_hr_list):
    pearson_per_signal = []
    HR_MAE = mae(np.array(predicted_hr_list), np.array(target_hr_list))
    HR_RMSE = rmse(np.array(predicted_hr_list), np.array(target_hr_list))

    # for (gt_signal, predicted_signal) in zip(target_hr_list, predicted_hr_list):
    #     r, p_value = pearsonr(predicted_signal, gt_signal)
    #     pearson_per_signal.append(r)

    # return {"MAE": np.mean(HR_MAE), "RMSE": HR_RMSE, "Pearson": np.mean(pearson_per_signal)}
    return {"MAE": np.mean(HR_MAE), "RMSE": HR_RMSE}


def run_training():
    print(sys.argv)
    
    norm_factor = 1

    dt = datetime.now()
    if config.USE_GT:
        save_path = config.PLOT_PATH + f'{config.DATASET}/{config.DATA_DIM} {config.MODEL} gt/{config.TARGET}/{dt.strftime("%Y_%m_%d_%H_%M_%S_%f")}/'
        save_path = str(Path(save_path).resolve())
        save_path += "/"
    else:
        save_path = config.PLOT_PATH + f'{config.DATASET}/{config.DATA_DIM} {config.MODEL} original/{config.TARGET}/{dt.strftime("%Y_%m_%d_%H_%M_%S_%f")}/'
        save_path = str(Path(save_path).resolve())
        save_path += "/"
    os.makedirs(save_path)

    # Model training feedback is handled by comet, change this based on your own account configuration
    experiment = Experiment(api_key="RskpClObaLigQ3GbZrWHTWG8m",
                            project_name="cwt-net",
                            workspace="bittnerma")

    experiment.add_tags(["lr: "+str(config.lr), "batch: "+str(config.BATCH_SIZE), "DATASET: "+config.DATASET,
                        "hr_path: "+config.TARGET_PATH, "data_path: "+config.TRAINSET, "model: "+config.MODEL])
    experiment.log_parameters(
        {
            "batch_size": config.BATCH_SIZE,
            "lr": config.lr,
            "DATASET": config.DATASET,
            "hr_path": config.TARGET_PATH,
            "data_path": config.TRAINSET,
            "model": config.MODEL
        }
    )
    experiment.log_asset("config.py")

    # check path to checkpoint directory
    if config.CHECKPOINT_PATH:
        if not os.path.exists(config.CHECKPOINT_PATH):
            os.makedirs(config.CHECKPOINT_PATH)
            print("Output directory is created")

    # --------------------------------------
    # Initialize Model
    # --------------------------------------

    if config.MODEL == 'resnet1d':
        model = ResNet1D(
            in_channels=1,
            base_filters=64,  # 64 for ResNet1D, 352 for ResNeXt1D
            kernel_size=16,
            stride=2,
            groups=1,
            n_block=10,
            n_classes=1,
            norm_factor=norm_factor,
            downsample_gap=2,
            increasefilter_gap=2,
            use_do=True,
            use_bn=True)
    elif config.MODEL == 'resnet2d':
        model = Resnet2D(config.DATA_DIM)
    elif config.MODEL == 'transformer1d':
        model = TransformerModel(seq_len=300, d_model=256, nhead=4, d_hid=2048, nlayers=8,
                                 norm_factor=norm_factor)
    elif config.MODEL == 'transformer2d':
        model = CWTNet2(config.MODEL, config.DATA_DIM)
    else:
        print("Invalid model type")

    if config.OPTIMIZER == 'noam':
        optimizer = NoamOpt(2, 500, torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)

    if torch.cuda.is_available():
        print('GPU available... using GPU')
        # torch.cuda.manual_seed_all(42)
    else:
        print("GPU not available, using CPU")

    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    model.to(device)

    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    #     optimizer, factor=0.8, patience=5, verbose=True
    # )
    loss_fn = nn.L1Loss()
    # loss_fn = nn.MSELoss()

    # Read from a pre-made csv file that contains data divided into folds for cross validation
    #video_files_train = np.load(config.TRAINSET)
    #video_files_test = np.load(config.TESTSET)
    #video_files_val = np.load(config.VALSET)

    if config.DATASET == "vipl":
        fold = config.VIPL_FOLDS[config.FOLD_NR]
        if config.DATA_DIM == '2d':
            files = glob.glob(config.DATA_PATH + "r*.csv")
            all_files = [config.DATA_PATH+re.split(r"\\|/",file)[-1][2:] for file in files]
            video_files_test = [config.DATA_PATH+re.split(r"\\|/",file)[-1][2:] for file in files if int(re.split(r"\\|/",file)[-1].split("_")[1][1:]) in fold[0]]
            video_files_val = [config.DATA_PATH+re.split(r"\\|/",file)[-1][2:] for file in files if int(re.split(r"\\|/",file)[-1].split("_")[1][1:]) in fold[1]]
        else:
            all_files = glob.glob(config.DATA_PATH + "*.npy")
            video_files_test = [file for file in all_files if int(re.split(r"\\|/",file)[-1].split("_")[0][1:]) in fold[0]]
            video_files_val = [file for file in all_files if int(re.split(r"\\|/",file)[-1].split("_")[0][1:]) in fold[1]]
    else:
        fold = config.VICAR_FOLDS[config.FOLD_NR]
        if config.DATA_DIM == '2d':
            files = glob.glob(config.DATA_PATH + "r*.csv")
            all_files = [config.DATA_PATH+re.split(r"\\|/",file)[-1][2:] for file in files]
            video_files_test = [config.DATA_PATH+re.split(r"\\|/",file)[-1][2:] for file in files if int(re.split(r"\\|/",file)[-1].split("_")[1]) in fold[0]]
            video_files_val = [config.DATA_PATH+re.split(r"\\|/",file)[-1][2:] for file in files if int(re.split(r"\\|/",file)[-1].split("_")[1]) in fold[1]]
        else:
            all_files = glob.glob(config.DATA_PATH + "*.npy")
            video_files_test = [file for file in all_files if int(re.split(r"\\|/",file)[-1].split("_")[0]) in fold[0]]
            video_files_val = [file for file in all_files if int(re.split(r"\\|/",file)[-1].split("_")[0]) in fold[1]]
    video_files_train = [file for file in all_files if file not in video_files_test and file not in video_files_val]

    video_files_train = np.array(video_files_train)
    video_files_test = np.array(video_files_test)
    video_files_val = np.array(video_files_val)
    print(f"Trainset: {len(video_files_train)}, Testset: {len(video_files_test)}, Trainset: {len(video_files_val)}")


    # Build Dataloaders
    if config.DATA_DIM == "3d":
        train_set = DataLoaderSTMaps(data_files=video_files_train, target_signal_path=config.TARGET_PATH)
        test_set = DataLoaderSTMaps(data_files=video_files_test, target_signal_path=config.TARGET_PATH)
        val_set = DataLoaderSTMaps(data_files=video_files_val, target_signal_path=config.TARGET_PATH)
    elif config.DATA_DIM == "2d":
        train_set = DataLoaderCWTNet(cwt_files=video_files_train, target_signal_path=config.TARGET_PATH)
        test_set = DataLoaderCWTNet(cwt_files=video_files_test, target_signal_path=config.TARGET_PATH)
        val_set = DataLoaderCWTNet(cwt_files=video_files_val, target_signal_path=config.TARGET_PATH)
    elif config.DATA_DIM == "1d":
        train_set = DataLoader1D(data_files=video_files_train, target_signal_path=config.TARGET_PATH)
        test_set = DataLoader1D(data_files=video_files_test, target_signal_path=config.TARGET_PATH)
        val_set = DataLoader1D(data_files=video_files_val, target_signal_path=config.TARGET_PATH)

    train_loader = torch.utils.data.DataLoader(
        dataset=train_set,
        batch_size=config.BATCH_SIZE,
        num_workers=config.NUM_WORKERS,
        shuffle=False,
        collate_fn=collate_fn
    )
    # Check if all data exists
    #for item in train_loader:
    #    pass

    print('\nTrain DataLoader constructed successfully!')

    test_loader = torch.utils.data.DataLoader(
        dataset=test_set,
        batch_size=config.BATCH_SIZE,
        num_workers=config.NUM_WORKERS,
        shuffle=False,
        collate_fn=collate_fn
    )
    # Check if all data exists
    #for item in test_loader:
    #    pass
    print('\nEvaluation DataLoader constructed successfully!')

    val_loader = torch.utils.data.DataLoader(
        dataset=val_set,
        batch_size=config.BATCH_SIZE,
        num_workers=config.NUM_WORKERS,
        shuffle=False,
        collate_fn=collate_fn
    )

    # Load checkpointed model (if  present)
    if config.USE_CHECKPOINT:
        model, optimizer, checkpointed_loss, checkpoint_flag = load_model_if_checkpointed(model, optimizer, config.CHECKPOINT_PATH, load_on_cpu=not torch.cuda.is_available())
        if checkpoint_flag:
            print(f"Checkpoint Found! Loading from checkpoint :: LOSS={checkpointed_loss}")
        else:
            print("Checkpoint Not Found! Training from beginning")
    else:
        print("Not using checkpoint! Training from beginning")
    checkpointed_loss = 0.0

    # --------------------------------------
    # Start training
    # --------------------------------------
    train_loss_per_epoch = []
    for epoch in range(config.EPOCHS):
        target_hr_list, predicted_hr_list, train_loss = train_eval_functions.train_fn(model, train_loader, optimizer, loss_fn)

        metrics = compute_criteria(target_hr_list, predicted_hr_list)

        print(f"\nFinished [Epoch: {epoch + 1}/{config.EPOCHS}]",
              "\nTraining Loss: {:.3f} |".format(train_loss),
              "HR_MAE : {:.3f} |".format(metrics["MAE"]),
              "HR_RMSE : {:.3f} |".format(metrics["RMSE"]),)
              # "Pearsonr : {:.3f} |".format(metrics["Pearson"]), )

        train_loss_per_epoch.append(train_loss)
        #writer.add_scalar("Loss/train", train_loss, epoch+1)

        # Plots on tensorboard
        #ba_plot_image = create_plot_for_tensorboard('bland_altman', target_hr_list, predicted_hr_list)
        #gtvsest_plot_image = create_plot_for_tensorboard('gt_vs_est', target_hr_list, predicted_hr_list)
        #writer.add_image('BA_plot', ba_plot_image, epoch)
        #writer.add_image('gtvsest_plot', gtvsest_plot_image, epoch)

        metrics = {
            'train_loss': train_loss,
        }
        experiment.log_metrics(metrics, epoch=epoch+1)

        ### Intermediate evaluation ###
        target_hr_list, predicted_hr_list, test_loss = train_eval_functions.eval_fn(model, test_loader, loss_fn)
        metrics = compute_criteria(target_hr_list, predicted_hr_list)
        metrics['test_loss'] = test_loss
        experiment.log_metrics(metrics, epoch=epoch + 1)

        # Save model with best test loss
        if checkpointed_loss != 0.0:
            if test_loss < checkpointed_loss:
                best_targets = target_hr_list
                best_predicted = predicted_hr_list
                save_model_checkpoint(model, optimizer, test_loss, save_path)
                os.remove(f"{save_path}running_model_{checkpointed_loss}.pt")
                checkpointed_loss = test_loss
        else:
            best_targets = target_hr_list
            best_predicted = predicted_hr_list
            save_model_checkpoint(model, optimizer, test_loss, save_path)
            checkpointed_loss = test_loss


    plot_train_curve(train_loss_per_epoch, save_path, target=config.TARGET)
    mean_loss = np.mean(train_loss_per_epoch)

    # Save the mean_loss value for each video instance to the writer
    print(f"Avg Training Loss: {np.mean(mean_loss)} for {config.EPOCHS} epochs")


    print(f"Finished Training, Validating {len(video_files_test)} video files for {config.EPOCHS_VAL} Epochs")
    checkpoint = torch.load(save_path + f"running_model_{checkpointed_loss}.pt")
    model.load_state_dict(checkpoint['model_state_dict'])
    for epoch in range(config.EPOCHS_VAL):
        target_hr_list, predicted_hr_list, val_loss = train_eval_functions.eval_fn(model, val_loader, loss_fn)
        # Compute testing metrics
        metrics = compute_criteria(target_hr_list, predicted_hr_list)
        pearson = np.corrcoef(target_hr_list, predicted_hr_list)[0,1]

        print(f"\nFinished evaluation",
              "\nValidation Loss: {:.3f} |".format(val_loss),
              "HR_MAE : {:.3f} |".format(metrics["MAE"]),
              "HR_RMSE : {:.3f} |".format(metrics["RMSE"]),
              "Pearson correlation : {:.3f} |".format(pearson))

        # Plots on the local storage.
        #gt_vs_est(target_hr_list, predicted_hr_list, plot_path=config.PLOT_PATH, loss=test_loss)
        #bland_altman_plot(target_hr_list, predicted_hr_list, plot_path=config.PLOT_PATH, loss=test_loss)
        #gt_vs_est(best_targets, best_predicted, target=sys.argv[1], plot_path=config.PLOT_PATH, loss=checkpointed_loss)
        #bland_altman_plot(best_targets, best_predicted, target=sys.argv[1], plot_path=config.PLOT_PATH, loss=checkpointed_loss)
        gt_vs_est(target_hr_list, predicted_hr_list, target=config.TARGET, plot_path=save_path, loss=val_loss)
        bland_altman_plot(target_hr_list, predicted_hr_list, target=config.TARGET, plot_path=save_path,
                          loss=val_loss)

        f = open(config.RESULTS_FILE, "a")
        if config.USE_GT:
            f.write(
                f"\n{config.DATASET} {config.DATA_DIM} {config.MODEL} gt {config.TARGET} {config.FOLD_NR}: MAE {str(metrics['MAE'])}, RMSE {str(metrics['RMSE'])}, P {str(pearson)}")
        else:
            f.write(
                f"\n{config.DATASET} {config.DATA_DIM} {config.MODEL} original {config.TARGET} {config.FOLD_NR}: MAE {str(metrics['MAE'])}, RMSE {str(metrics['RMSE'])}, P {str(pearson)}")
        f.close()

        print("done")




if __name__ == '__main__':
    run_training()