from pipelines import train_decoder_only, train_generate_validate_pipeline

if __name__ == "__main__":
    # Configuration de base
    base_config = {
        'dataset_path': "out/markov_python_generated_dataset10000.csv",
        'seed': 42,
        'batch_size': 512,
        'val_split': 0.8,
        'vocab_size': 17,
        'padding_idx': 0,
        'n_head': 4,
        'd_model': 32,
        'nb_layers': 15,
        'lr': 5e-5,
        'nb_epoch': 200,
        'dim_feedforward': 1024,
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
        'auto_precision': False  # Nouveau paramètre pour activer/désactiver torch.cuda.amp
    }
    #Essai baseline
    # print("=== Essai baseline ===")
    # validator_baseline = train_generate_validate_pipeline(base_config)

    # Essai cyclical learning rate
    config_cyclical = base_config.copy()
    config_cyclical["scheduler"] = {
        "name": "cyclical",
        "params": {"base_lr": 5e-8, "max_lr": 5e-5, "step_size_up": 782}
    }

    print("=== Essai cyclical learning rate ===")
    validator_cyclical = train_generate_validate_pipeline(config_cyclical)

    # Essai de scheduler learning rate (ReduceOnPlatau)
    config_plateau = base_config.copy()
    config_plateau["scheduler"] = {
        "name": "ReduceOnPlatau",
        "params": {"mode": "min", "factor": 0.1, "patience": 3}
    }
    print("=== Essai de scheduler learning rate (ReduceOnPlatau) ===")
    validator_plateau = train_generate_validate_pipeline(config_plateau)

    #Essai de early stopping
    config_early = base_config.copy()
    config_early["scheduler"] = {
        "name": "None",
        "params": {}
    }
    config_early["early_stopping"] = {
        "name": "patience",
        "params": {"patience": 20, "verbose": True, "delta": 0.0005}
    }
    print("=== Essai de early stopping ===")
    validator_early = train_generate_validate_pipeline(config_early)

    #Essai de précision auto (précision adaptative)
    config_auto_precision = base_config.copy()
    config_auto_precision["scheduler"] = {
        "name": "None",
        "params": {}
    }
    config_auto_precision["early_stopping"] = {"name": "None", "params": {}}
    config_auto_precision["auto_precision"] = True
    print("=== Essai de précision auto ===")
    validator_auto_precision = train_generate_validate_pipeline(config_auto_precision)

