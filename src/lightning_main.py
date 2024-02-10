import comet_ml
import pytorch_lightning as pl
import torchmetrics
import torch.nn as nn
import torch
from pytorch_lightning.loggers import CometLogger, CSVLogger
from model_factory import get_model
from dataloader_factory import get_dataloaders
from dataloader_factory import get_data_path,get_name_to_id_func
from models.PeakbasedDetector import PeakbasedDetector
from omegaconf import OmegaConf
from constants import DataFoldsNew
from pathlib import Path
import pandas as pd


class Runner(pl.LightningModule):
    def __init__(self, cfg, model):
        super().__init__()
        self.cfg = cfg
        self.model = model
        self.loss_fn = nn.L1Loss()
        
        self.train_mae = torchmetrics.MeanAbsoluteError()        
        self.val_mae = torchmetrics.MeanAbsoluteError()        
        self.test_mae = torchmetrics.MeanAbsoluteError() 
        # self.automatic_optimization = True      
        

    def forward(self, x):
        # Runner needs to redirect any model.forward() calls to the actual
        # network
        return self.model(x)

    def configure_optimizers(self):
        if self.cfg.optimize.optimizer == 'Adam':
            optimizer = torch.optim.Adam(self.model.parameters(), lr=self.cfg.optimize.lr)
        elif self.cfg.optimize.optimizer == 'SGD':
            optimizer = torch.optim.SGD(self.model.parameters(), lr=self.cfg.optimize.lr)
        else:
            raise NotImplementedError(f"Optimizer {self.cfg.optimizer}")
        
        if self.cfg.optimize.scheduler:
            scheduler = getattr(torch.optim.lr_scheduler,self.cfg.optimize.scheduler)(optimizer, **self.cfg.optimize.scheduler_settings)
            # return {"optimizer":optimizer,"scheduler":scheduler,"metric":"train-MAE"}
            return [optimizer],[{"scheduler":scheduler,"monitor":"train-MAE"}]
            
        
        return optimizer

    def _step(self, batch):        
        # inputs = torch.stack(tuple(data['data'] for data in batch))
        # targets = torch.stack(tuple(data['target'] for data in batch))
        inputs = batch['data']
        targets = batch['target']
        if 'time' in batch.keys():
            time = batch['time']
        # print(targets)
        # if torch.isclose(targets,torch.tensor(0).float()).any():
            # print(f"Empty target detected in {batch['name']}")
        if isinstance(self.model,PeakbasedDetector):
            outputs = self.model(inputs,time)
        else:
            outputs = self.model(inputs)
        loss = self.loss_fn(outputs, targets.view(outputs.shape))

        return loss, targets, outputs
        

    def training_step(self, batch, batch_idx):
        loss, y, y_hat = self._step(batch)
        self.train_mae(y_hat, y.view(y_hat.shape))

        # Log step-level loss & accuracy
        self.log("train/loss_step", loss,batch_size=self.cfg.train.batch_size)        
        return loss

    def validation_step(self, batch, batch_idx):
        loss, y, y_hat = self._step(batch)
        self.val_mae(y_hat, y.view(y_hat.shape))

        # Log step-level loss & accuracy
        self.log("val/loss_step", loss,batch_size=self.cfg.train.batch_size)
        return loss

    def test_step(self, batch, batch_idx):
        loss, y, y_hat = self._step(batch)
        self.test_mae(y_hat, y.view(y_hat.shape))

        # Log test loss
        self.log("test/loss_step", loss,batch_size=self.cfg.train.batch_size)
        for i,name in enumerate(batch['name']):
            self.log("test/target"+f"/{name}", y[i].detach().cpu(),batch_size=self.cfg.train.batch_size)
            self.log("test/pred"+f"/{name}", y_hat.flatten()[i].detach().cpu(),batch_size=self.cfg.train.batch_size)
        return loss

    def on_train_epoch_end(self):
        # Log the epoch-level training accuracy
        self.log('train-MAE', self.train_mae.compute())   
        print(f"Learning Rate: {self.optimizers().param_groups[0]['lr']}")
        self.train_mae.reset()

    def on_validation_epoch_end(self):
        # Log the epoch-level validation accuracy
        self.log('val-MAE', self.val_mae.compute())
        self.val_mae.reset()

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

if __name__ == "__main__":

    # Load defaults and overwrite by command-line arguments
    cfg = OmegaConf.load("x_config.yaml")
    cmd_cfg = OmegaConf.from_cli()
    cfg = OmegaConf.merge(cfg, cmd_cfg)

    data_cfg = OmegaConf.load("x_dataset_config.yaml")
    data_cfg = data_cfg[cfg.dataset.name]

    # if config_exists_in_project(cfg):
    #     print("This analysis has already been done exiting")
    #     exit()
    
    print("Training Configuration")
    print(OmegaConf.to_yaml(cfg))

    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    print(f"Using device: {device}")    

    # Seed everything. Note that this does not make training entirely
    # deterministic.
    pl.seed_everything(cfg.seed, workers=True)

    comet_logger = CometLogger(**cfg.comet) 
    comet_logger.log_hyperparams(OmegaConf.to_container(cfg,resolve=True))
    experiment_name = cfg.comet.project_name + "/" + comet_logger.experiment.name
    csv_logger = CSVLogger("csv_logs",name=experiment_name)

    model = get_model(cfg.model.name,cfg.model.data_dim,data_cfg.traces_fps,list(cfg.model.target))
    model = model.to(device)
    
    runner = Runner(cfg, model)

    checkpoint_dir = Path(__file__).parent.parent / 'model_checkpoints' / experiment_name
    
    if not checkpoint_dir.exists():
        checkpoint_dir.mkdir(parents=True)

    OmegaConf.save(cfg,checkpoint_dir / "config.yaml" )

    # cfg = OmegaConf.to_object(cfg)

    training_callbacks = initialize_callbacks(cfg,checkpoint_dir=checkpoint_dir)

    trainer = pl.Trainer(
        max_epochs=cfg.train.epochs,
        logger=[comet_logger,csv_logger],               
        accelerator='auto',
        callbacks=training_callbacks,
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
                                                          target=list(cfg.model.target),
                                                          input_representation=cfg.model.input_representation,
                                                          test_ids=test_ids,
                                                          val_ids=val_ids,
                                                          device=device,
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
        test_df[['Set','Type','ID']] = test_df['index'].str.split("/",expand=True)
        test_df = test_df.drop(columns=['index']).rename(columns={0:'value'})
        test_df['Target'] = cfg.model.target
        test_df['Model'] = cfg.model.name
        test_df['InptRep'] = cfg.model.input_representation
        test_df['Dataset'] = cfg.dataset.name        
        test_df['GT'] = int(cfg.dataset.use_gt)
        test_df['Fold'] = int(cfg.dataset.fold_number)
        test_df.to_csv(checkpoint_dir / "TestResults.csv",index=False)