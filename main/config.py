import numpy as np
import os
import multiprocessing
from main.image_encoder_selection import IMAGE_ENCODER_SELECTION

class Config(object):
    NAME = None  # Override in sub-classes
    MODE = 'train'  # Mode (train/eval)
    ROOT_DIR = None  # Root project directory

    LEARNING_RATE = 1e-4
    BATCH_SIZE = 16
    NUM_EPOCHS = 150
    CHECKPOINT_PATH = os.path.join('checkpoints')
    DISPLAY_EVERY = 20
    SAVE_EVERY = 5

    BACKBONE = {IMAGE_ENCODER_SELECTION: 2048}
    UNITS = 512
    EMBED_SIZE = 512
    VOCAB_SIZE = None
    WINDOW_SIZE = 30

    DATASET_PATH = os.path.join('datasets')  # Override in sub-classes
    MAXLEN = 10
    BUFFER_SIZE = 1000

    FREQUENCY = None
    USE_BIAS_VECTOR = True
    START_WORD = '<sos>'
    END_WORD = '<eos>'
    UNK_WORD = None

    def __init__(self):
        if os.name == 'nt':
            self.WORKERS = 0
        else:
            self.WORKERS = multiprocessing.cpu_count()

        assert self.MODE in ['train', 'test']

    def display(self):
        print("\nConfigurations:")
        print("-" * 30)
        for attr in dir(self):
            if not attr.startswith("__") and not callable(getattr(self, attr)):
                print("{:30} {}".format(attr, getattr(self, attr)))
        print()
