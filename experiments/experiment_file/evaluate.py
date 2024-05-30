import os
import glob
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

class EvaluationConfig(Config):
    NAME = 'v2c_IIT-V2C'
    MODE = 'test'
    ROOT_DIR = PROJECT_ROOT
    CHECKPOINT_PATH = os.path.join(PROJECT_ROOT, 'checkpoints')
    DATASET_PATH = os.path.join(PROJECT_ROOT, 'datasets', 'IIT-V2C')
    MAXLEN = 10

config = EvaluationConfig()
vocab = pickle.load(open(os.path.join(config.CHECKPOINT_PATH, 'vocab.pkl'), 'rb'))
annotation_file = config.MODE + '.txt'
clips, targets, _, config = load.parse_dataset(config, annotation_file, vocab=vocab)
test_dataset = load.FeatureDataset(clips, targets)
test_loader = data.DataLoader(test_dataset, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=config.WORKERS)
config.display()

v2c_model = Video2Command(config)
v2c_model.build()

if not os.path.exists(os.path.join(config.CHECKPOINT_PATH, 'prediction')):
    os.makedirs(os.path.join(config.CHECKPOINT_PATH, 'prediction'))

checkpoint_files = sorted(glob.glob(os.path.join(config.CHECKPOINT_PATH, 'saved', '*.pth')))
for checkpoint_file in checkpoint_files:
    epoch = int(checkpoint_file.split('_')[-1][:-4])
    v2c_model.load_weights(checkpoint_file)
    predicted, actual = v2c_model.evaluate(test_loader, vocab)

    with open(os.path.join(config.CHECKPOINT_PATH, 'prediction', 'prediction_{}.txt'.format(epoch)), 'w') as f:
        for i in range(len(predicted)):
            predicted_command = utils.sequence_to_text(predicted[i], vocab)
            actual_command = utils.sequence_to_text(actual[i], vocab)
            f.write('------------------------------------------\n')
            f.write(str(i) + '\n')
            f.write(predicted_command + '\n')
            f.write(actual_command + '\n')

    print('Ready for cococaption.')
