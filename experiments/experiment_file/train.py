import os
import sys
import pickle

import numpy as np
import torch
from torch.utils import data

PROJECT_ROOT = os.path.abspath("../../")

sys.path.append(PROJECT_ROOT)
from main.model import *
from main import utils
from main.config import *
from datasets import load

class TrainingConfig(Config):
    NAME = 'v2c_IIT-V2C'
    MODE = 'train'
    ROOT_DIR = PROJECT_ROOT
    CHECKPOINT_PATH = os.path.join(PROJECT_ROOT, 'checkpoints')
    DATASET_PATH = os.path.join(PROJECT_ROOT, 'datasets', 'IIT-V2C')
    MAXLEN = 10

config = TrainingConfig()
annotation_file = config.MODE + '.txt'
clips, targets, vocab, config = load.parse_dataset(config, annotation_file)
config.display()
train_dataset = load.FeatureDataset(clips, targets)
train_loader = data.DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=config.WORKERS)
bias_vector = vocab.get_bias_vector() if config.USE_BIAS_VECTOR else None

v2c_model = Video2Command(config)
v2c_model.build(bias_vector)

with open(os.path.join(config.CHECKPOINT_PATH, 'vocab.pkl'), 'wb') as f:
    pickle.dump(vocab, f)

v2c_model.train(train_loader)
