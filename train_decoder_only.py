"""
Training Script for Decoder-Only Transformer Models.

This script allows for both testing and full-scale training of decoder-only Transformer models.
It supports various configurations, including baseline, cyclical learning rate, and early stopping.

Modules:
    - pipelines: Contains the training and validation pipeline functions.

Execution:
    Run this script directly to start training. Use the `test` variable to toggle between test and full modes.
"""

from pipelines import train_decoder_only, train_generate_validate_pipeline

if __name__ == "__main__":
    # Variable pour contrôler si on fait un test ou une version complète
    test = True  # Mettre à False pour la version complète

    if test:
        # Configuration de base pour test
        configs = [(15, 32, 1024)]  # Uniquement la première configuration pour le test
        nb_epochs = 1  # Réduit à 1 epoch pour le test
        sync = True
        print("=== Mode Test ===")
    else:
        # Configuration de base complète
        configs = [
            # Configuration de base avec d_model = 32
            (15, 32, 1024),  # NBL-15_DM-32_DFF-1024
            (27, 32, 128),   # NBL-27_DM-32_DFF-128
            (27, 32, 256),   # NBL-27_DM-32_DFF-256
            (8, 32, 2048),   # NBL-8_DM-32_DFF-2048

            # Configuration avec d_model = 64
            (15, 64, 1024),  # NBL-15_DM-64_DFF-1024
            (27, 64, 128),   # NBL-27_DM-64_DFF-128
            (27, 64, 256),   # NBL-27_DM-64_DFF-256
            (8, 64, 2048),   # NBL-8_DM-64_DFF-2048

            # # Configuration avec d_model = 128
            (15, 128, 1024),  # NBL-15_DM-128_DFF-1024
            # (27, 128, 128),   # NBL-27_DM-128_DFF-128
            # (27, 128, 256),   # NBL-27_DM-128_DFF-256
            # (8, 128, 2048),   # NBL-8_DM-128_DFF-2048
        ]
        nb_epochs = 500
        sync = True
        print("=== Mode Complet ===")

    for nb_layers, d_model, dim_feedforward in configs:
        config_dict = {
            'dataset_path': "out/markov_python_generated_dataset10000.csv",
            'seed': 42,
            'batch_size': 512,
            'val_split': 0.8,
            'vocab_size': 17,
            'padding_idx': 0,
            'n_head': 4,
            'd_model': d_model,
            'nb_layers': nb_layers,
            'lr': 5e-5,
            'nb_epoch': nb_epochs,
            'dim_feedforward': dim_feedforward,
            'dynamic': True,
            'scheduler': {
                'name': 'None',
                'params': {}
            },
            'early_stopping': {
                'name': 'None',
                'params': {}
            },
            'continue_training': True,
            'checkpoint_path': "model_state.pth",
            'auto_precision': True,
            'graph_name': f"DO_NBL-{nb_layers}_DM-{d_model}_DFF-{dim_feedforward}_baseline"
        }
        #Essai baseline
        print(f"=== {'Test' if test else 'Essai'} baseline ===")
        validator_baseline = train_generate_validate_pipeline(config_dict,sync_wandb=sync)

        # Essai cyclical learning rate
        config_cyclical = config_dict.copy()
        config_cyclical["scheduler"] = {
            "name": "cyclical",
            "params": {"base_lr": 5e-8, "max_lr": 5e-4, "step_size_up": 782}
        }
        config_cyclical["graph_name"] = f"DO_NBL-{nb_layers}_DM-{d_model}_DFF-{dim_feedforward}_cyclical"

        print(f"=== {'Test' if test else 'Essai'} cyclical learning rate ===")
        validator_cyclical = train_generate_validate_pipeline(config_cyclical,sync_wandb=sync)

        #Essai de early stopping
        config_early = config_dict.copy()
        config_early["scheduler"] = {
            "name": "None",
            "params": {}
        }
        config_early["early_stopping"] = {
            "name": "patience",
            "params": {"patience": 100, "verbose": True, "delta": 0.005}
        }
        config_early["graph_name"] = f"DO_NBL-{nb_layers}_DM-{d_model}_DFF-{dim_feedforward}_early"
        print(f"=== {'Test' if test else 'Essai'} early stopping ===")
        validator_early = train_generate_validate_pipeline(config_early,sync_wandb=sync)
