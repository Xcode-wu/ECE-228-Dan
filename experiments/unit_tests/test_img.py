import os
import sys

import numpy as np
import torch
from torch.utils import data
import torchvision.transforms as transforms

# Root directory of the project
ROOT_DIR = os.path.abspath("../../")

# Import v2c utils
sys.path.append(ROOT_DIR)  # To find local version of the library
from main.model import *
from datasets import load
from main import utils
from main.config import *

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
    
# Test configuration
config = TrainConfig()
config.display()
print()

# Test parse_dataset
annotation_file = config.MODE + '.txt'
clips, targets, vocab, config = load.parse_dataset(config, annotation_file, numpy_features=False)
config.display()

print('Vocabulary:')
print(vocab.word2idx)
print('length ("<pad>" included):', len(vocab))
print('dataset:', len(clips), len(targets))
print()

transform = transforms.Compose([transforms.Resize(224), 
                                transforms.ToTensor()])

train_dataset = load.FeatureDataset(clips, 
                                       targets, 
                                       numpy_features=False, 
                                       transform=transform)
# Test torch dataloader object
for i, (Xv, S, clip_name) in enumerate(train_dataset):
    print(Xv.shape, S.shape, clip_name)
    break
