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
        self.do_regression = cfg.train.do_regression
        self.do_classification = cfg.train.do_classification
        
        self.loss_fn = getattr(nn,cfg.train.loss)() #nn.MSELoss() #nn.L1Loss()        
        
        self.cls_loss_fn = getattr(nn,cfg.train.classification_loss)()

        self.train_mae = torchmetrics.MeanAbsoluteError()        
        self.val_mae = torchmetrics.MeanAbsoluteError()        
        self.test_mae = torchmetrics.MeanAbsoluteError() 

        self.train_acc = torchmetrics.Accuracy(task='multiclass',num_classes=10)        
        self.val_acc = torchmetrics.Accuracy(task='multiclass',num_classes=10)        
        self.test_acc = torchmetrics.Accuracy(task='multiclass',num_classes=10) 
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
        targets = batch['regression_target']
        if 'time' in batch.keys():
            time = batch['time']
        # print(targets)
        # if torch.isclose(targets,torch.tensor(0).float()).any():
            # print(f"Empty target detected in {batch['name']}")
        if isinstance(self.model,PeakbasedDetector):
            outputs = self.model(inputs,time)
        else:
            outputs = self.model(inputs)

        reg_pred,cls_pred = outputs
        
        loss = {}

        if self.do_regression:
            reg_loss = self.loss_fn(reg_pred, targets.view(reg_pred.shape))
            loss['regression_loss'] = reg_loss
        
        if self.do_classification:
            cls_loss = self.cls_loss_fn(cls_pred, batch['classification_target'].to(torch.long))
            loss['classification_loss'] = cls_loss            

        return loss, (targets,batch['classification_target']), outputs
        

    def training_step(self, batch, batch_idx):
        loss, y, y_hat = self._step(batch)

        if self.do_regression:
            ## Multiple losses means classification and regression
            self.train_mae(y_hat[0], y[0].view(y_hat[0].shape))
            self.log("train/reg_loss_step", loss['regression_loss'],batch_size=batch['data'].shape[0])                    
            
        if self.do_classification:
            self.train_acc(y_hat[1], y[1])
            # Log step-level loss & accuracy
            self.log("train/cls_loss_step", loss['classification_loss'],batch_size=batch['data'].shape[0])        
        
        loss = torch.sum(torch.stack(list(loss.values())))
        self.log("train/loss_step", loss,batch_size=batch['data'].shape[0])                    
  
        return loss

    def validation_step(self, batch, batch_idx):
        loss, y, y_hat = self._step(batch)        

        if self.do_regression:
            ## Multiple losses means classification and regression
            self.val_mae(y_hat[0], y[0].view(y_hat[0].shape))
            self.log("val/reg_loss_step", loss['regression_loss'],batch_size=batch['data'].shape[0])                    
            
        if self.do_classification:
            self.val_acc(y_hat[1], y[1])
            # Log step-level loss & accuracy
            self.log("val/cls_loss_step", loss['classification_loss'],batch_size=batch['data'].shape[0])        
        
        loss = torch.sum(torch.stack(list(loss.values())))
        self.log("val/loss_step", loss,batch_size=batch['data'].shape[0])    
        return loss

    def test_step(self, batch, batch_idx):
        loss, y, y_hat = self._step(batch)                

        if self.do_regression:
            ## Multiple losses means classification and regression
            self.test_mae(y_hat[0], y[0].view(y_hat[0].shape))
            self.log("test/reg_loss_step", loss['regression_loss'],batch_size=batch['data'].shape[0])                    

            for i,name in enumerate(batch['name']):
                if not isinstance(self.cfg.model.target,str):
                    for j,target_name in enumerate(self.cfg.model.target):
                        self.log("test/target"+f"/{name}/{target_name}", y[0][i,j].detach().cpu(),batch_size=self.cfg.train.batch_size)
                        self.log("test/pred"+f"/{name}/{target_name}", y_hat[0][i,j].detach().cpu(),batch_size=self.cfg.train.batch_size)
                else:
                    self.log("test/target"+f"/{name}", y[0][i].detach().cpu(),batch_size=self.cfg.train.batch_size)
                    self.log("test/pred"+f"/{name}", y_hat[0].flatten()[i].detach().cpu(),batch_size=self.cfg.train.batch_size)
            
        if self.do_classification:
            self.test_acc(y_hat[1], y[1])
            # Log step-level loss & accuracy
            self.log("test/cls_loss_step", loss['classification_loss'],batch_size=batch['data'].shape[0])        
        
        loss = torch.sum(torch.stack(list(loss.values())))
        self.log("test/loss_step", loss,batch_size=batch['data'].shape[0])    

        return loss

    def on_train_epoch_end(self):
        # Log the epoch-level training accuracy
        self.log('train-MAE', self.train_mae.compute())   
        self.log('train-ACC', self.train_acc.compute())   
        # print(f"Learning Rate: {self.optimizers().param_groups[0]['lr']}")
        self.train_mae.reset()
        self.train_acc.reset()

    def on_validation_epoch_end(self):
        # Log the epoch-level validation accuracy
        self.log('val-MAE', self.val_mae.compute())
        self.log('val-ACC', self.val_acc.compute())
        self.val_mae.reset()
        self.val_acc.reset()

    def on_test_epoch_end(self):
        # Log the epoch-level validation accuracy
        self.log('test-MAE', self.test_mae.compute())
        self.test_mae.reset()