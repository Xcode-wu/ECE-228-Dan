import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms

PROJECT_ROOT = os.path.abspath("../../")

sys.path.append(PROJECT_ROOT)
import datasets.load as load
from main.config import *
from main.model import *

class FeatureExtractionConfig(Config):
    NAME = 'Feature_Extraction'
    MODE = 'test'
    ROOT_DIR = PROJECT_ROOT
    DATASET_PATH = os.path.join(PROJECT_ROOT, 'datasets', 'IIT-V2C')
    WINDOW_SIZE = 30
    BATCH_SIZE = 50

def extract_features(dataset_dir, dataset, model_name):
    output_dir = os.path.join(dataset_dir, model_name)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print('Loading pre-trained model...')
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = CNNWrapper(backbone=model_name, checkpoint_path=os.path.join(PROJECT_ROOT, 'checkpoints', 'backbone', 'resnet50.pth'))
    model.eval()
    model.to(device)
    print('Model loaded.')

    for i, (Xv, S, clip_name) in enumerate(dataset):
        with torch.no_grad():
            Xv = Xv.to(device)
            print('-' * 30)
            print('Processing clip {}...'.format(clip_name))
            outputs = model(Xv)
            outputs = outputs.view(outputs.shape[0], -1)
            output_file = os.path.join(output_dir, clip_name + '.npy')
            np.save(output_file, outputs.cpu().numpy())
            print('{}: {}'.format(clip_name + '.npy', S))
            print('Shape: {}, saved to {}.'.format(outputs.shape, output_file))
    del model
    return

def main():
    config = FeatureExtractionConfig()
    model_names = ['resnet50']

    annotation_files = ['train.txt', 'test.txt']
    for annotation_file in annotation_files:
        annotations = load.load_annotations(config.DATASET_PATH, annotation_file)
        clips, targets, vocab, config = load.parse_dataset(config, annotation_file, numpy_features=False)
        config.display()
        transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        image_dataset = load.FeatureDataset(clips, targets, numpy_features=False, transform=transform)

        for model_name in model_names:
            extract_features(config.DATASET_PATH, image_dataset, model_name)

if __name__ == '__main__':
    main()
