"""
parameters.py

This module stores all the configurable parameters and hyperparameters used
across the project, ensuring easy tuning and reproducibility.

Author: Gavin Gao
Date: May 2025
"""

params = {
    'net_type': 'Autoencoder',
    # data parameters
    'dev_dir': '/root/CODE/DCASE2020_asd/dcase2020_task2_datasets/dev_data',
    'eval_dir': '/root/CODE/DCASE2020_asd/dcase2020_task2_datasets/eval_data',
    'feat_dir': 'features',

    'log_dir': 'logs',
    'checkpoints_dir': 'checkpoints',
    'result_dir': 'result',
    'result_file': 'result.csv',
    'target': 'fan',
    
    # audio feature extraction parameters
    'sample_rate': 16000,
    'n_fft': 1024,
    'hop_length': 512,
    'n_mels': 128,
    'power': 2.0,
    'n_frames': 5,
    

    # training parameters
    'batch_size': 512,
    'epochs': 100,
    'learning_rate': 0.001,
    'b1': 0.9,
    'b2': 0.999,
    'weight_decay': 0.0,
    'shuffle': True,
    'num_workers': 8,
    'pin_memory': True,
    'dropout': 0.1,
    'max_grad_norm': 1.0,
    'seed': 42,
    'validation_split': 0.1,
    'verbose': True,
    'early_stopping_patience': 10,
}

