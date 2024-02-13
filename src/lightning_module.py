import pytorch_lightning as pl
import torchmetrics
import torch.nn as nn
import torch

from .models.PeakbasedDetector import PeakbasedDetector

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
            if not isinstance(self.cfg.model.target,str):
                for j,target_name in enumerate(self.cfg.model.target):
                    self.log("test/target"+f"/{name}/{target_name}", y[i,j].detach().cpu(),batch_size=self.cfg.train.batch_size)
                    self.log("test/pred"+f"/{name}/{target_name}", y_hat[i,j].detach().cpu(),batch_size=self.cfg.train.batch_size)
            else:
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