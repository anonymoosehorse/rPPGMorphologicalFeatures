from pathlib import Path

import comet_ml
import pytorch_lightning as pl
import torch
from pytorch_lightning.loggers import CometLogger, CSVLogger
from omegaconf import OmegaConf,ListConfig
import pandas as pd

from .model_factory import get_model
from .dataloader_factory import get_dataloaders
from .dataloader_factory import get_data_path,get_name_to_id_func
from .models.PeakbasedDetector import PeakbasedDetector
from .constants import DataFoldsNew
from .lightning_module import Runner


def config_exists_in_project(cfg):
    cfgs_in_project = list((Path("model_checkpoints") / cfg.comet.project_name).rglob("*.yaml"))
    containered_cfg = OmegaConf.to_container(cfg,resolve=True)
    for cfg_path in cfgs_in_project:
        test_cfg = OmegaConf.load(cfg_path)
        if containered_cfg == OmegaConf.to_container(test_cfg,resolve=True):
            return True
        
    return False

def initialize_callbacks(cfg,checkpoint_dir):
    training_callbacks = []
    
    # Create an instance of ModelCheckpoint
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        monitor='val-MAE',  # Metric to monitor for best performance
        dirpath=checkpoint_dir,  # Directory where checkpoints will be saved
        filename=r'{epoch}-{val-MAE:.2f}',  # File name prefix for saved models
        save_top_k=1,  # Save only the best model
        mode='min',  # 'min' or 'max' depending on the monitored metric
    )

    training_callbacks.append(checkpoint_callback)

    if cfg.train.early_stopping:
        
        # early_stopping_cb = pl.callbacks.EarlyStopping(
        #     monitor="val-MAE",
        #     min_delta=0.01,
        #     patience=3,
        #     mode='min'
        # )
        early_stopping_cb = pl.callbacks.EarlyStopping(
            **cfg.train.early_stopping_settings
        )
        training_callbacks.append(early_stopping_cb)

    training_callbacks.append(pl.callbacks.LearningRateMonitor(logging_interval="epoch"))

    return training_callbacks

def run_training(checkpoint_dir,cfg,data_cfg,experiment_name=None,debug_mode=False):

    if config_exists_in_project(cfg):
        print("This analysis has already been done exiting")
        return
    
    print("Training Configuration")
    print(OmegaConf.to_yaml(cfg))


    # Seed everything. Note that this does not make training entirely
    # deterministic.
    pl.seed_everything(cfg.seed, workers=True)
    
    loggers = None
    
    if not debug_mode:
        comet_logger = CometLogger(**cfg.comet) 
        comet_logger.log_hyperparams(OmegaConf.to_container(cfg,resolve=True))
        if experiment_name is None:
            experiment_name = cfg.comet.project_name + "/" + comet_logger.experiment.name
        
        csv_logger = CSVLogger("csv_logs",name=experiment_name)

        loggers = [comet_logger,csv_logger]

        checkpoint_dir = checkpoint_dir / experiment_name

    model = get_model(cfg.model.name,cfg.model.data_dim,data_cfg.traces_fps,list(cfg.model.target) if isinstance(cfg.model.target,ListConfig) else cfg.model.target)    
    
    runner = Runner(cfg, model)    

    if not debug_mode:
        
        if not checkpoint_dir.exists():
            checkpoint_dir.mkdir(parents=True)

        OmegaConf.save(cfg,checkpoint_dir / "config.yaml" )

    # cfg = OmegaConf.to_object(cfg)
        
    training_callbacks = None

    if not debug_mode:
        training_callbacks = initialize_callbacks(cfg,checkpoint_dir=checkpoint_dir)

    trainer = pl.Trainer(
        max_epochs=cfg.train.epochs,
        logger=loggers,                       
        accelerator='auto' if not debug_mode else 'cpu',
        callbacks=training_callbacks,
        log_every_n_steps=2,
        deterministic=True
    )
    
    # data = get_data(cfg.dataset.root,cfg.dataset.name,cfg.model.input_representation,cfg.dataset.use_gt)    
    data_path = get_data_path(cfg.dataset.root,cfg.dataset.name,cfg.model.input_representation,cfg.dataset.use_gt)    

    loader_settings = {
        "batch_size":cfg.train.batch_size,
        "num_workers":cfg.dataset.num_workers,
        "shuffle":cfg.train.shuffle
    }

    data_folds = DataFoldsNew(cfg.dataset.use_gt,cfg.dataset.name)
    test_ids,val_ids = data_folds.get_fold(cfg.dataset.fold_number)

    # train_loader,test_loader,val_loader = get_dataloaders(cfg,device)

    train_loader,test_loader,val_loader = get_dataloaders(data_path=data_path,
                                                          target=list(cfg.model.target) if isinstance(cfg.model.target,ListConfig) else cfg.model.target,
                                                          input_representation=cfg.model.input_representation,
                                                          test_ids=test_ids,
                                                          val_ids=val_ids,                                                          
                                                          name_to_id_func=get_name_to_id_func(cfg.dataset.name),
                                                          normalize_data=cfg.dataset.normalize_data,
                                                          flip_signal=cfg.dataset.flip_signal,
                                                          **loader_settings)
    checkpoint_path = None
    if not cfg.model.name == "peakdetection1d":
        trainer.fit(runner, train_loader, test_loader)

    
        ## Load model with the lowest validation score
        checkpoint_path = list(checkpoint_dir.glob('*.ckpt'))[0]
        print(f"Using Checkpoint file {checkpoint_path} for testing")

    # Test (if test dataset is implemented)
    if val_loader is not None:        
        test_results = trainer.test(runner,ckpt_path=checkpoint_path, dataloaders=val_loader)
        
        test_df = pd.DataFrame(test_results).T.reset_index().iloc[1:]
        if isinstance(cfg.model.target,ListConfig):
            test_df[['Set','Type','ID','Target']] = test_df['index'].str.split("/",expand=True)
        else:
            test_df[['Set','Type','ID']] = test_df['index'].str.split("/",expand=True)
            test_df['Target'] = cfg.model.target
        test_df = test_df.drop(columns=['index']).rename(columns={0:'value'})
        test_df['Model'] = cfg.model.name
        test_df['InptRep'] = cfg.model.input_representation
        test_df['Dataset'] = cfg.dataset.name        
        test_df['GT'] = int(cfg.dataset.use_gt)
        test_df['Fold'] = int(cfg.dataset.fold_number)
        test_df.to_csv(checkpoint_dir / "TestResults.csv",index=False)

if __name__ == "__main__":
    checkpoint_dir = Path(__file__).parent.parent / 'model_checkpoints'
    config_path = Path(__file__).parent.parent / "x_config.yaml"
    dataset_config_path = Path(__file__).parent.parent / "x_dataset_config.yaml"
    run_training(checkpoint_dir,config_path,dataset_config_path)