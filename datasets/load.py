import os
import sys
import glob
import pickle

import numpy as np
from PIL import Image
import torch
from torch.utils import data

ROOT_DIR = os.path.abspath("../")
sys.path.append(ROOT_DIR)
import main.utils as utils

def load_data(data_path=os.path.join('datasets', 'IIT-V2C'),
              data_file='train.txt'):
    def frame_range(start_frame, end_frame):
        return list(range(start_frame, end_frame + 1))

    annotations = {}
    with open(os.path.join(data_path, data_file), 'r') as f:
        i = 0
        lines = []
        for line in f:
            line = line.strip()
            i += 1
            lines.append(line)

            if i % 4 == 0:
                video_name, clip_id = '_'.join(lines[0].split('_')[:-1]), lines[0].split('_')[-1]
                start_frame, end_frame = int(lines[1].split(' ')[0]), int(lines[1].split(' ')[1])
                frames = frame_range(start_frame, end_frame)
                command = lines[2].strip().split(' ')

                if video_name not in annotations:
                    annotations[video_name] = [[clip_id, frames, command]]
                else:
                    annotations[video_name].append([clip_id, frames, command])

                lines = []

    return annotations

def data_summary(data):
    total_clips = 0
    total_frames = 0
    for video in data.keys():
        video_annotations = data[video]
        total_clips += len(video_annotations)
        for annotation in video_annotations:
            total_frames += len(annotation[1])
    print('# videos in total:', len(data))
    print('# sub-video clips:', total_clips)
    print('# frames in total:', total_frames)

def extract_clips_captions(data, add_padding=True):
    clips, captions = [], []
    for video in data.keys():
        video_annotations = data[video]
        for annotation in video_annotations:
            clip_name = video + '_' + annotation[0]
            if add_padding:
                caption = '<sos> ' + ' '.join(annotation[2]) + ' <eos>'
            else:
                caption = ' '.join(annotation[2])
            clips.append(clip_name)
            captions.append(caption)
    return clips, captions

def convert_videos_to_frames(data_path, in_folder='avi_video', out_folder='images'):
    import cv2
    video_paths = glob.glob(os.path.join(data_path, in_folder, '*.avi'))
    for video_path in video_paths:
        video_name = video_path.strip().split('/')[-1][:-4]
        save_path = os.path.join(data_path, out_folder, video_name)
        print('Saving:', save_path)
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        cap = cv2.VideoCapture(video_path)
        success, frame = cap.read()
        count = 0
        cap.set(cv2.CAP_PROP_FPS, 15)
        fps = cap.get(cv2.CAP_PROP_FPS)
        print("FPS: {}".format(fps))

        while success:
            cv2.imwrite(os.path.join(save_path, 'frame%d.png' % count), frame)
            count += 1
            success, frame = cap.read()
    return True

def prepare_image_paths(data, max_frames=30, data_path=os.path.join('datasets', 'IIT-V2C'), folder='images', synthetic_frame=os.path.join('imagenet_frame.png'), add_padding=True):
    def get_image_paths(frame_numbers, video_name, max_frames=30, data_path=os.path.join('datasets', 'IIT-V2C'), folder='images', synthetic_frame=os.path.join('imagenet_frame.png')):
        num_frames = len(frame_numbers)
        loop_factor = min(num_frames, max_frames)

        paths = []
        for i in range(loop_factor):
            img_path = os.path.join(data_path, folder, video_name, 'frame{}.png'.format(frame_numbers[i]))
            if os.path.isfile(img_path):
                paths.append(img_path)

        while len(paths) < max_frames:
            paths.append(synthetic_frame)

        return paths

    inputs, targets = [], []
    for video in data.keys():
        video_annotations = data[video]
        for annotation in video_annotations:
            clip_name = video + '_' + annotation[0]
            frames = get_image_paths(annotation[1], video, max_frames, data_path, folder, synthetic_frame)
            if add_padding:
                target = '<sos> ' + ' '.join(annotation[2]) + ' <eos>'
            else:
                target = ' '.join(annotation[2])
            inputs.append({clip_name: frames})
            targets.append(target)

    return inputs, targets

def process_dataset(config, data_file, vocab=None, use_numpy=True):
    annotations = load_data(config.DATASET_PATH, data_file)

    if use_numpy:
        clips, captions = extract_clips_captions(annotations)
        clips = [os.path.join(config.DATASET_PATH, list(config.BACKBONE.keys())[0], x + '.npy') for x in clips]
    else:
        clips, captions = prepare_image_paths(annotations, max_frames=config.WINDOW_SIZE, data_path=config.DATASET_PATH, folder='images', synthetic_frame=os.path.join(config.ROOT_DIR, 'datasets', 'imagenet_frame.png'))

    if vocab is None:
        vocab = utils.build_vocab(captions, frequency=config.FREQUENCY, start_word=config.START_WORD, end_word=config.END_WORD, unk_word=config.UNK_WORD)
    config.VOCAB_SIZE = len(vocab)

    if config.MAXLEN is None:
        config.MAXLEN = utils.get_maxlen(captions)

    targets = utils.texts_to_sequences(captions, vocab)
    targets = utils.pad_sequences(targets, config.MAXLEN, padding='post')
    targets = targets.astype(np.int64)

    return clips, targets, vocab, config

class CustomDataset(data.Dataset):
    def __init__(self, inputs, targets, use_numpy=True, transform=None):
        self.inputs = inputs
        self.targets = targets
        self.use_numpy = use_numpy
        self.transform = transform

    def _load_image(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            img = img.convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        return img

    def _parse_clip(self, clip):
        frames = []
        clip_name = list(clip.keys())[0]
        paths = clip[clip_name]
        for path in paths:
            img = self._load_image(path)
            frames.append(img)
        frames = torch.stack(frames, dim=0)
        return frames, clip_name

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        if self.use_numpy:
            X = np.load(self.inputs[idx])
            clip_name = self.inputs[idx].split('/')[-1]
        else:
            X, clip_name = self._parse_clip(self.inputs[idx])
        Y = self.targets[idx]
        return X, Y, clip_name
