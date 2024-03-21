from omegaconf import OmegaConf
import torch
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from matplotlib import pyplot as plt

from .dataloader_factory import get_dataloaders
from .dataloader_factory import get_data_path,get_name_to_id_func
from .constants import DataFoldsNew
from .models.PeakbasedDetector import PeakbasedDetector

def run_peakbased_detector(cfg:OmegaConf,dataset_cfg:OmegaConf,out_dir_name:str):
    device = torch.device('cpu') #torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    print(f"Using device: {device}")   

    data_folds = DataFoldsNew(cfg.dataset.use_gt,cfg.dataset.name)
    test_ids,val_ids = data_folds.get_fold(cfg.dataset.fold_number)

    data_path = get_data_path(cfg.dataset.root,cfg.dataset.name,cfg.model.input_representation,cfg.dataset.use_gt)    

    loader_settings = {
        "batch_size":cfg.train.batch_size,
        "num_workers":cfg.dataset.num_workers,
        "shuffle":False
    }

    train_loader,test_loader,val_loader = get_dataloaders(data_path=data_path,
                                                            target=list(cfg.model.target),
                                                            input_representation='traces',
                                                            test_ids=test_ids,
                                                            val_ids=val_ids,                                                            
                                                            name_to_id_func=get_name_to_id_func(cfg.dataset.name),
                                                            normalize_data=cfg.dataset.normalize_data,
                                                            flip_signal=cfg.dataset.flip_signal,
                                                            **loader_settings)
    
    output_dir = Path(__file__).parent.parent / out_dir_name
    output_dir.mkdir(exist_ok=True)

    out_file_name = f"{cfg.dataset.name}_{cfg.dataset.fold_number}"
    if cfg.dataset.use_gt:
        out_file_name += "_gt"
    out_file_name += ".csv"

    out_file_path = output_dir / out_file_name

    # if out_file_path.exists():
    #     print(f"File {out_file_path} already exists")
    #     exit()

    with torch.no_grad():

        

        model = PeakbasedDetector(list(cfg.model.target),30)
        
        df_list = []
        for batch in tqdm(val_loader):
            
            est = model(batch['data'],batch['time'])

            # plt.plot(batch['gt_data'][0][0,:],batch['gt_data'][0][1,:],label="GT_Signal")
            # plt.plot(batch['gt_data'][0][0,:],batch['gt_data'][0][2,:],label="GT_Peaks")
            # plt.legend()

            # plt.show()
            tar_df = pd.DataFrame(batch['regression_target'],columns=list(cfg.model.target)).assign(ID=batch['name']).melt(var_name='Target',id_vars=['ID'])
            tar_df['Type'] = 'target'
            
            est_df = pd.DataFrame(est).assign(ID=batch['name']).melt(var_name='Target',id_vars=['ID'])
            est_df['Type'] = 'pred'
            
            df = pd.concat([tar_df,est_df])
            df_list.append(df)
        df = pd.concat(df_list)
        
        df['Model'] = cfg.model.name
        df['InptRep'] = cfg.model.input_representation
        df['Dataset'] = cfg.dataset.name        
        df['GT'] = int(cfg.dataset.use_gt)
        df['Fold'] = int(cfg.dataset.fold_number)
        
        df.to_csv(out_file_path,index=False)

if __name__ == "__main__":
    cfg = OmegaConf.load("x_config.yaml")
    cmd_cfg = OmegaConf.from_cli()
    cfg = OmegaConf.merge(cfg, cmd_cfg)

    data_cfg = OmegaConf.load("x_dataset_config.yaml")
    data_cfg = data_cfg[cfg.dataset.name]

    run_peakbased_detector(cfg,data_cfg,"BaselineAnalysisNewData")