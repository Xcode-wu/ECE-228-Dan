import os
import sys
import glob
import pickle

import numpy as np
from PIL import Image
import torch
from torch.utils import data

PROJECT_DIR = os.path.abspath("../")

sys.path.append(PROJECT_DIR)
import main.utils as utils

def load_annotations(dataset_dir=os.path.join('datasets', 'IIT-V2C'), annotation_file='train.txt'):
    def generate_frames(start_frame, end_frame):
        return [i for i in range(start_frame, end_frame + 1)]

    annotations = {}
    with open(os.path.join(dataset_dir, annotation_file), 'r') as f:
        i = 0
        annotation = []
        for line in f:
            line = line.strip()
            i += 1
            annotation.append(line)

            if i % 4 == 0:
                video_file, clip_id = '_'.join(annotation[0].split('_')[:-1]), annotation[0].split('_')[-1]
                start_frame, end_frame = int(annotation[1].split(' ')[0]), int(annotation[1].split(' ')[1])
                frames = generate_frames(start_frame, end_frame)
                command = annotation[2].strip().split(' ')

                if video_file not in annotations:
                    annotations[video_file] = [[clip_id, frames, command]]
                else:
                    annotations[video_file].append([clip_id, frames, command])

                annotation = []

    return annotations

def dataset_summary(annotations):
    num_clips = 0
    total_frames = 0
    for video in annotations.keys():
        clips = annotations[video]
        num_clips += len(clips)
        for clip in clips:
            total_frames += len(clip[1])
    print('# videos in total:', len(annotations))
    print('# sub-video clips in annotation file:', num_clips)
    print('# frames in total:', total_frames)

def extract_clips_and_commands(annotations, add_padding=True):
    clip_names, commands = [], []
    for video_file in annotations.keys():
        clips = annotations[video_file]
        for clip in clips:
            clip_name = video_file + '_' + clip[0]
            command = '<sos> ' + ' '.join(clip[2]) + ' <eos>' if add_padding else ' '.join(clip[2])
            commands.append(command)
            clip_names.append(clip_name)

    return clip_names, commands

def convert_videos_to_images(dataset_dir, input_folder='avi_video', output_folder='images'):
    import cv2
    video_paths = glob.glob(os.path.join(dataset_dir, input_folder, '*.avi'))
    for video_path in video_paths:
        video_file = video_path.strip().split('/')[-1][:-4]
        save_path = os.path.join(dataset_dir, output_folder, video_file)
        print('Saving:', save_path)
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        cap = cv2.VideoCapture(video_path)
        success, frame = cap.read()
        frame_count = 0
        cap.set(cv2.CAP_PROP_FPS, 15)
        fps = cap.get(cv2.CAP_PROP_FPS)
        print("Frames per second using video.get(cv2.CAP_PROP_FPS) : {0}".format(fps))

        while success:
            cv2.imwrite(os.path.join(save_path, 'frame%d.png' % frame_count), frame)
            frame_count += 1
            success, frame = cap.read()
    return True

def generate_image_paths_and_commands(annotations, max_frames=30, dataset_dir=os.path.join('datasets', 'IIT-V2C'), image_folder='images', placeholder_image=os.path.join('imagenet_frame.png'), add_padding=True):
    def get_image_paths(frame_numbers, video_file, max_frames=30, dataset_dir=os.path.join('datasets', 'IIT-V2C'), image_folder='images', placeholder_image=os.path.join('imagenet_frame.png')):
        loop_factor = min(len(frame_numbers), max_frames)
        image_paths = [os.path.join(dataset_dir, image_folder, video_file, 'frame{}.png'.format(frame_numbers[i])) for i in range(loop_factor) if os.path.isfile(os.path.join(dataset_dir, image_folder, video_file, 'frame{}.png'.format(frame_numbers[i])))]

        while len(image_paths) < max_frames:
            image_paths.append(placeholder_image)

        return image_paths

    inputs, targets = [], []
    for video_file in annotations.keys():
        clips = annotations[video_file]
        for clip in clips:
            clip_name = video_file + '_' + clip[0]
            frames_path = get_image_paths(clip[1], video_file, max_frames, dataset_dir, image_folder, placeholder_image)
            command = '<sos> ' + ' '.join(clip[2]) + ' <eos>' if add_padding else ' '.join(clip[2])
            inputs.append({clip_name: frames_path})
            targets.append(command)

    return inputs, targets

def parse_dataset(config, annotation_file, vocab=None, use_numpy_features=True):
    annotations = load_annotations(config.DATASET_PATH, annotation_file)

    if use_numpy_features:
        clips, captions = extract_clips_and_commands(annotations)
        clips = [os.path.join(config.DATASET_PATH, list(config.BACKBONE.keys())[0], x + '.npy') for x in clips]
    else:
        clips, captions = generate_image_paths_and_commands(annotations, max_frames=config.WINDOW_SIZE, dataset_dir=config.DATASET_PATH, image_folder='images', placeholder_image=os.path.join(config.ROOT_DIR, 'datasets', 'imagenet_frame.png'))

    if vocab is None:
        vocab = utils.build_vocab(captions, frequency=config.FREQUENCY, start_word=config.START_WORD, end_word=config.END_WORD, unk_word=config.UNK_WORD)
    
    config.VOCAB_SIZE = len(vocab)

    if config.MAXLEN is None:
        config.MAXLEN = utils.get_maxlen(captions)
    
    targets = utils.texts_to_sequences(captions, vocab)
    targets = utils.pad_sequences(targets, config.MAXLEN, padding='post')
    targets = targets.astype(np.int64)

    return clips, targets, vocab, config

class FeatureDataset(data.Dataset):
    def __init__(self, inputs, targets, use_numpy_features=True, transform=None):
        self.inputs, self.targets = inputs, targets
        self.use_numpy_features = use_numpy_features
        self.transform = transform

    def parse_clip(self, clip):
        images = []
        clip_name = list(clip.keys())[0]
        image_paths = clip[clip_name]
        for image_path in image_paths:
            img = self.read_image(image_path)
            images.append(img)
        images = torch.stack(images, dim=0)
        return images, clip_name

    def read_image(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            img = img.convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        return img

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        if self.use_numpy_features:
            features = np.load(self.inputs[idx])
            clip_name = self.inputs[idx].split('/')[-1]
        else:
            features, clip_name = self.parse_clip(self.inputs[idx])
        target_sequence = self.targets[idx]
        return features, target_sequence, clip_name
