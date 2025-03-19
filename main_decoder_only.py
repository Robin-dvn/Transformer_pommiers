from pipelines import train_decoder_only, train_generate_validate_pipeline


if __name__ == "__main__":
    # Define the configuration dictionary
    config_dict = {
        'dataset_path': "out/markov_python_generated_dataset10000.csv",
        'seed': 42,
        'batch_size': 512,
        'val_split': 0.8,
        'vocab_size': 17,
        'padding_idx': 0,
        'n_head': 2,
        'd_model': 8,
        'nb_layers': 15,
        'lr': 5e-3,
        'nb_epoch': 1,
        'dim_feedforward': 1024,
        'dynamic': True,
        'cyclical': False,
        'continue_training': False,
        'checkpoint_path': None
    }

    couples_ffl = [(32,1)]
    for ff, layers in couples_ffl:
        config_dict['dim_feedforward'] = ff
        config_dict['nb_layers'] = layers

        # Validate the model using the pipeline
        validator = train_generate_validate_pipeline(config_dict)