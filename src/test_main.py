from omegaconf import OmegaConf

if __name__ == "__main__":

    # Load defaults and overwrite by command-line arguments
    cfg = OmegaConf.load("x_config.yaml")
    cmd_cfg = OmegaConf.from_cli()
    cfg = OmegaConf.merge(cfg, cmd_cfg)
    
    print("Training Configuration")
    print(OmegaConf.to_yaml(cfg))