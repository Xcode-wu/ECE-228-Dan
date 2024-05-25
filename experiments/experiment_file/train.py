import os
import sys
import pickle

import numpy as np
import torch
from torch.utils import data

# Root directory of the project
ROOT_DIR = os.path.abspath("../../")

# Import v2c utils
sys.path.append(ROOT_DIR)  # To find local version of the library
from main.model import *
from main import utils
from main.config import *
from datasets import load

# Configuration for hperparameters
class TrainConfig(Config):
    """Configuration for training with IIT-V2C.
    """
    NAME = 'v2c_IIT-V2C'
    MODE = 'train'
    ROOT_DIR = ROOT_DIR
    CHECKPOINT_PATH = os.path.join(ROOT_DIR, 'checkpoints')
    DATASET_PATH = os.path.join(ROOT_DIR, 'datasets', 'IIT-V2C')
    MAXLEN = 10

# Setup configuration class
config = TrainConfig()
# Setup tf.dataset object
annotation_file = config.MODE + '.txt'
clips, targets, vocab, config = load.parse_dataset(config, annotation_file)
config.display()
train_dataset = load.FeatureDataset(clips, targets)
train_loader = data.DataLoader(train_dataset, 
                               batch_size=config.BATCH_SIZE, 
                               shuffle=True, 
                               num_workers=config.WORKERS)
bias_vector = vocab.get_bias_vector() if config.USE_BIAS_VECTOR else None

# Setup and build video2command training inference
v2c_model = Video2Command(config)
v2c_model.build(bias_vector)

# Save vocabulary at last
with open(os.path.join(config.CHECKPOINT_PATH, 'vocab.pkl'), 'wb') as f:
    pickle.dump(vocab, f)

# Start training
v2c_model.train(train_loader)