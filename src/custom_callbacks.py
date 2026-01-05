from pytorch_lightning.callbacks import Callback
import pandas as pd
import torch

import pytorch_lightning as pl
import pandas as pd
import torch

class PerSampleTestLogger(pl.Callback):
    def __init__(self, cfg, save_path="TestResults.csv"):
        super().__init__()
        self.cfg = cfg
        self.save_path = save_path
        self.results = []

    def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        # Assumes batch contains: 'name', 'data', targets in y[0], predictions in y_hat[0]
        with torch.no_grad():
            # Forward pass to get predictions (your model should handle batch and return tuple)
            loss, y, y_hat = pl_module._step(batch)
            # y = batch['target'] if 'target' in batch else batch['y']  # Adjust this if needed

            # Classification or regression? (uses your flag)
            if pl_module.do_regression:
                # Multiple targets
                if not isinstance(self.cfg.model.target, str):
                    for i, name in enumerate(batch['name']):
                        for j, target_name in enumerate(self.cfg.model.target):
                            self.results.append({
                                "Set": "test",
                                "Type": "pred",
                                "ID": name,
                                "Target": target_name,
                                "value": y_hat[0][i, j].detach().cpu().item()
                            })
                            self.results.append({
                                "Set": "test",
                                "Type": "target",
                                "ID": name,
                                "Target": target_name,
                                "value": y[0][i, j].detach().cpu().item()
                            })
                # Single target
                else:
                    for i, name in enumerate(batch['name']):
                        self.results.append({
                            "Set": "test",
                            "Type": "pred",
                            "ID": name,
                            "Target": self.cfg.model.target,
                            "value": y_hat[0].flatten()[i].detach().cpu().item()
                        })
                        self.results.append({
                            "Set": "test",
                            "Type": "target",
                            "ID": name,
                            "Target": self.cfg.model.target,
                            "value": y[0][i].detach().cpu().item()
                        })
            # If you also have a classification branch, you could extend here

    def on_test_epoch_end(self, trainer, pl_module):
        if self.results:
            df = pd.DataFrame(self.results)
            # Add model and dataset metadata as columns (single value per column)
            df['Model'] = self.cfg.model.name
            df['InptRep'] = self.cfg.model.input_representation
            df['Dataset'] = self.cfg.dataset.name        
            df['GT'] = int(self.cfg.dataset.use_gt)
            df['Fold'] = int(self.cfg.dataset.fold_number)
            df.to_csv(self.save_path, index=False)
            print(f"Saved per-sample test results to {self.save_path}")
            self.results = []

            # Optional: log to Comet.ml as an artifact if logger is present
            if trainer.logger and hasattr(trainer.logger.experiment, "log_asset"):
                trainer.logger.experiment.log_asset(self.save_path)
