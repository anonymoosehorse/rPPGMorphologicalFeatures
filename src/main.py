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

from utils.plot_scripts import plot_train_curve, bland_altman_plot, gt_vs_est
from utils.model_utils import load_model_if_checkpointed, save_model_checkpoint

from model_factory import get_model
from dataloader_factory import get_dataloaders

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
    
    

    dt = datetime.now()
    if config.USE_GT:
        save_path = config.PLOT_PATH / f'{config.DATASET}/{config.DATA_DIM} {config.MODEL} gt/{config.TARGET}/{dt.strftime("%Y_%m_%d_%H_%M_%S_%f")}/'        
    else:
        save_path = config.PLOT_PATH / f'{config.DATASET}/{config.DATA_DIM} {config.MODEL} original/{config.TARGET}/{dt.strftime("%Y_%m_%d_%H_%M_%S_%f")}/'
        
    os.makedirs(save_path)

    # Model training feedback is handled by comet, change this based on your own account configuration
    experiment = Experiment(api_key="YourAPIKeyHere",
                            project_name="cwt-net",
                            workspace="YourWorkSpaceHere")

    experiment.add_tags([f"lr: {config.lr}", f"batch: {config.BATCH_SIZE}", f"DATASET: {config.DATASET}",
                        f"hr_path: {config.TARGET_PATH}", f"data_path: {config.TRAINSET}", f"model: {config.MODEL}"])
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

    model = get_model(config.MODEL,config.DATA_DIM)
    
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

    train_loader,test_loader,val_loader = get_dataloaders(config)
    
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


    print(f"Finished Training, Validating {len(train_loader)} video files for {config.EPOCHS_VAL} Epochs")
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