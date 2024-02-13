
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