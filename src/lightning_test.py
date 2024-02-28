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

def run_test(cfg_path: str):    
    
    cfg = OmegaConf.load(cfg_path / "config.yaml")

    data_cfg = OmegaConf.load("x_dataset_config.yaml")
    data_cfg = data_cfg[cfg.dataset.name]

    # Seed everything. Note that this does not make training entirely
    # deterministic.
    pl.seed_everything(cfg.seed, workers=True)    

    model = get_model(cfg.model.name,cfg.model.data_dim,data_cfg.traces_fps,list(cfg.model.target) if isinstance(cfg.model.target,ListConfig) else cfg.model.target)
    
    runner = Runner(cfg, model)
    
    trainer = pl.Trainer(
        max_epochs=cfg.train.epochs,                       
        accelerator='auto',        
        log_every_n_steps=2
    )
    
    # data = get_data(cfg.dataset.root,cfg.dataset.name,cfg.model.input_representation,cfg.dataset.use_gt)    
    data_path = get_data_path(cfg.dataset.root,cfg.dataset.name,cfg.model.input_representation,cfg.dataset.use_gt)    

    loader_settings = {
        "batch_size":cfg.train.batch_size,
        "num_workers":cfg.dataset.num_workers,
        "shuffle":False
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
        
    ## Load model with the lowest validation score
    checkpoint_path = list(cfg_path.glob('*.ckpt'))[0]        

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
        test_df.to_csv(cfg_path / "TestResults.csv",index=False)

if __name__ == "__main__":

    run_test()
    # Load defaults and overwrite by command-line arguments

    