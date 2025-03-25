from pipelines import train_decoder_only, train_generate_validate_pipeline

if __name__ == "__main__":
    # Variable pour contrôler si on fait un test ou une version complète
    test = True  # Mettre à False pour la version complète

    if test:
        # Configuration de base pour test
        couple_l_ff = [(15, 1024)]  # Uniquement le premier couple pour le test
        nb_epochs = 100  # Réduit à 50 epochs pour le test
        print("=== Mode Test ===")
    else:
        # Configuration de base complète
        couple_l_ff = [(15, 1024), (27, 128), (27, 256), (8, 2048)]
        nb_epochs = 500
        print("=== Mode Complet ===")

    for layers, ff in couple_l_ff:
        base_config = {
            'dataset_path': "out/markov_python_generated_dataset10000.csv",
            'seed': 42,
            'batch_size': 512,
            'val_split': 0.8,
            'vocab_size': 17,
            'padding_idx': 0,
            'n_head': 4,
            'd_model': 32,
            'nb_layers': layers,
            'lr': 5e-5,
            'nb_epoch': nb_epochs,
            'dim_feedforward': ff,
            'dynamic': True,
            'scheduler': {
                'name': 'None',
                'params': {}
            },
            'early_stopping': {
                'name': 'None',
                'params': {}
            },
            'continue_training': False,
            'checkpoint_path': None,
            'auto_precision': True,
            'graph_name': f"DO_NBL-{layers}_DM-32_DFF-{ff}_baseline"
        }
        #Essai baseline
        print(f"=== {'Test' if test else 'Essai'} baseline ===")
        validator_baseline = train_generate_validate_pipeline(base_config,sync_wandb=True)

        # # Essai cyclical learning rate
        # config_cyclical = base_config.copy()
        # config_cyclical["scheduler"] = {
        #     "name": "cyclical",
        #     "params": {"base_lr": 5e-8, "max_lr": 5e-5, "step_size_up": 782}
        # }
        # config_cyclical["graph_name"] = f"DO_NBL-{layers}_DM-32_DFF-{ff}_cyclical"

        # print(f"=== {'Test' if test else 'Essai'} cyclical learning rate ===")
        # validator_cyclical = train_generate_validate_pipeline(config_cyclical)

        # #Essai de early stopping
        # config_early = base_config.copy()
        # config_early["scheduler"] = {
        #     "name": "None",
        #     "params": {}
        # }
        # config_early["early_stopping"] = {
        #     "name": "patience",
        #     "params": {"patience": 100, "verbose": True, "delta": 0.001}
        # }
        # config_early["graph_name"] = f"DO_NBL-{layers}_DM-32_DFF-{ff}_early"
        # print(f"=== {'Test' if test else 'Essai'} early stopping ===")
        # validator_early = train_generate_validate_pipeline(config_early)

        # #Essai de précision auto (précision adaptative)
        # config_auto_precision = base_config.copy()
        # config_auto_precision["scheduler"] = {
        #     "name": "None",
        #     "params": {}
        # }
        # config_auto_precision["early_stopping"] = {"name": "None", "params": {}}
        # config_auto_precision["auto_precision"] = True
        # config_auto_precision["graph_name"] = f"DO_NBL-{layers}_DM-32_DFF-{ff}_auto_precision"
        # print("=== Essai de précision auto ===")
        # validator_auto_precision = train_generate_validate_pipeline(config_auto_precision)

