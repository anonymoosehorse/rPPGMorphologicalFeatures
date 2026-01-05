import pytorch_lightning as pl
import torchmetrics
import torch.nn as nn
import torch
from omegaconf import ListConfig

from .losses import CCCLoss, InverseCorrelationLoss
from .models.PeakbasedDetector import PeakbasedDetector

class Runner(pl.LightningModule):

    def try_get_loss(self, loss_name):
        try:
            loss = getattr(nn,loss_name)()
            return loss
        except AttributeError:
            try:
                if loss_name == "CCCLoss":
                    return CCCLoss
                elif loss_name == "InverseCorrelationLoss":
                    return InverseCorrelationLoss
            except AttributeError:
                raise ValueError(f"Unknown loss function: {loss_name}")
        else:
            raise ValueError(f"Unknown loss function: {loss_name}")

    def __init__(self, cfg, model):
        super().__init__()
        self.cfg = cfg
        self.model = model
        self.do_regression = cfg.train.do_regression
        self.do_classification = cfg.train.do_classification

        if isinstance(cfg.train.loss,(list,ListConfig)):
            self.loss_fn = [self.try_get_loss(l) for l in cfg.train.loss]
        else:
            self.loss_fn = self.try_get_loss(cfg.train.loss)

        self.cls_loss_fn = self.try_get_loss(cfg.train.classification_loss)

        self.train_mae = torchmetrics.MeanAbsoluteError()        
        self.val_mae = torchmetrics.MeanAbsoluteError()        
        self.test_mae = torchmetrics.MeanAbsoluteError() 

        self.train_pearson = torchmetrics.PearsonCorrCoef()
        self.val_pearson = torchmetrics.PearsonCorrCoef()
        self.test_pearson = torchmetrics.PearsonCorrCoef()

        self.train_ccc = torchmetrics.ConcordanceCorrCoef()
        self.val_ccc = torchmetrics.ConcordanceCorrCoef()
        self.test_ccc = torchmetrics.ConcordanceCorrCoef()

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
            return [optimizer],[{"scheduler":scheduler,"monitor":"train/loss_epoch"}]
            
        
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
            if isinstance(self.loss_fn,list):
                reg_loss = sum(l(reg_pred, targets.view(reg_pred.shape)) for l in self.loss_fn)
            else:
                reg_loss = self.loss_fn(reg_pred, targets.view(reg_pred.shape))
            # corr_loss = 1 - torch.corrcoef(torch.stack([reg_pred,targets.view(reg_pred.shape)]).squeeze())[0,1]
            # corr_loss = self.ccc_loss(reg_pred, targets.view(reg_pred.shape))
            loss['regression_loss'] = reg_loss
        
        if self.do_classification:
            cls_loss = self.cls_loss_fn(cls_pred, batch['classification_target'].to(torch.long))
            loss['classification_loss'] = cls_loss            

        return loss, (targets,batch['classification_target']), outputs
        

    def training_step(self, batch, batch_idx):
        loss, y, y_hat = self._step(batch)

        if self.do_regression:
            ## Multiple losses means classification and regression
            pred = y_hat[0].squeeze().detach()
            targ = y[0].detach().reshape_as(pred)

            self.train_mae.update(pred, targ)
            self.train_ccc.update(pred * 100, targ * 100)
            self.train_pearson.update(pred * 100, targ * 100)
            self.log("train/reg_loss", loss['regression_loss'],batch_size=batch['data'].shape[0],on_step=True,on_epoch=True)                    
            
        if self.do_classification:
            self.train_acc.update(y_hat[1], y[1])
            # Log step-level loss & accuracy
            self.log("train/cls_loss", loss['classification_loss'],batch_size=batch['data'].shape[0],on_step=True,on_epoch=True)        
        
        loss = torch.sum(torch.stack(list(loss.values())))
        self.log("train/loss", loss,batch_size=batch['data'].shape[0],on_step=True,on_epoch=True)        
        self.log("train/MAE", self.train_mae, batch_size=batch['data'].shape[0], on_step=False, on_epoch=True)
        self.log("train/ACC", self.train_acc, batch_size=batch['data'].shape[0], on_step=False, on_epoch=True)     
        self.log("train/CCC", self.train_ccc, batch_size=batch['data'].shape[0], on_step=False, on_epoch=True)
        self.log("train/Pearson", self.train_pearson, batch_size=batch['data'].shape[0], on_step=False, on_epoch=True)

        return loss

    def validation_step(self, batch, batch_idx):
        loss, y, y_hat = self._step(batch)        

        if self.do_regression:
            ## Multiple losses means classification and regression
            pred = y_hat[0].squeeze().detach()
            targ = y[0].detach().reshape_as(pred)
            
            self.val_ccc.update(pred.double()*100, targ.double()*100)       # more stable in fp64
            self.val_pearson.update(pred.double()*100, targ.double()*100)   # <<< key change
            self.log("val/reg_loss", loss['regression_loss'],batch_size=batch['data'].shape[0],on_step=True,on_epoch=True)                    
            
        if self.do_classification:
            self.val_acc.update(y_hat[1], y[1])
            # Log step-level loss & accuracy
            self.log("val/cls_loss", loss['classification_loss'],batch_size=batch['data'].shape[0],on_step=True,on_epoch=True)        
        
        loss = torch.sum(torch.stack(list(loss.values())))
        self.log("val/loss", loss,batch_size=batch['data'].shape[0],on_step=True,on_epoch=True)    
        self.log("val/MAE", self.val_mae, batch_size=batch['data'].shape[0], on_step=False, on_epoch=True)
        self.log("val/ACC", self.val_acc, batch_size=batch['data'].shape[0], on_step=False, on_epoch=True)
        self.log("val/CCC", self.val_ccc, batch_size=batch['data'].shape[0], on_step=False, on_epoch=True,sync_dist=True)
        self.log("val/Pearson", self.val_pearson, batch_size=batch['data'].shape[0], on_step=False, on_epoch=True,sync_dist=True)
        return loss

    def test_step(self, batch, batch_idx):
        loss, y, y_hat = self._step(batch)                

        if self.do_regression:
            pred = y_hat[0].squeeze().detach()
            targ = y[0].detach().reshape_as(pred)
            
            self.test_ccc.update(pred.double()*100, targ.double()*100)       # more stable in fp64
            self.test_pearson.update(pred.double()*100, targ.double()*100)   # <<< key change
            ## Multiple losses means classification and regression
            self.test_mae.update(y_hat[0], y[0].view(y_hat[0].shape))
            self.log("test/reg_loss", loss['regression_loss'],batch_size=batch['data'].shape[0],on_step=True,on_epoch=True)                    

            # for i,name in enumerate(batch['name']):
            #     if not isinstance(self.cfg.model.target,str):
            #         for j,target_name in enumerate(self.cfg.model.target):
            #             self.log("test/target"+f"/{name}/{target_name}", y[0][i,j].detach().cpu(),batch_size=self.cfg.train.batch_size)
            #             self.log("test/pred"+f"/{name}/{target_name}", y_hat[0][i,j].detach().cpu(),batch_size=self.cfg.train.batch_size)
            #     else:
            #         self.log("test/target"+f"/{name}", y[0][i].detach().cpu(),batch_size=self.cfg.train.batch_size)
            #         self.log("test/pred"+f"/{name}", y_hat[0].flatten()[i].detach().cpu(),batch_size=self.cfg.train.batch_size)
            
        if self.do_classification:
            self.test_acc.update(y_hat[1], y[1])
            # Log step-level loss & accuracy
            self.log("test/cls_loss", loss['classification_loss'],batch_size=batch['data'].shape[0],on_step=True,on_epoch=True)        
        
        loss = torch.sum(torch.stack(list(loss.values())))
        self.log("test/loss", loss,batch_size=batch['data'].shape[0],on_step=True,on_epoch=True)    
        self.log("test/MAE", self.test_mae, batch_size=batch['data'].shape[0], on_step=False, on_epoch=True)
        self.log("test/ACC", self.test_acc, batch_size=batch['data'].shape[0], on_step=False, on_epoch=True)
        self.log("test/CCC", self.test_ccc, batch_size=batch['data'].shape[0], on_step=False, on_epoch=True,sync_dist=True)
        self.log("test/Pearson", self.test_pearson, batch_size=batch['data'].shape[0], on_step=False, on_epoch=True,sync_dist=True)

        return loss

