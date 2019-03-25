import os
import sys
import numpy as np
import soundfile
import librosa
import h5py
import math
import pandas as pd
from sklearn import metrics
import logging
import matplotlib.pyplot as plt

import config


def create_folder(fd):
    if not os.path.exists(fd):
        os.makedirs(fd)


def get_filename(path):
    path = os.path.realpath(path)
    name_ext = path.split('/')[-1]
    name = os.path.splitext(name_ext)[0]
    return name


def create_logging(log_dir, filemode):
    create_folder(log_dir)
    i1 = 0

    while os.path.isfile(os.path.join(log_dir, '{:04d}.log'.format(i1))):
        i1 += 1
        
    log_path = os.path.join(log_dir, '{:04d}.log'.format(i1))
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
        datefmt='%a, %d %b %Y %H:%M:%S',
        filename=log_path,
        filemode=filemode)

    # Print to console
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(name)-12s: %(levelname)-8s %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)
    
    return logging
    
    
def read_audio(audio_path, target_fs=None):
    (audio, fs) = soundfile.read(audio_path)

    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)
        
    if target_fs is not None and fs != target_fs:
        audio = librosa.resample(audio, orig_sr=fs, target_sr=target_fs)
        fs = target_fs
        
    return audio, fs
    
    
def pad_truncate_sequence(x, max_len):
    if len(x) < max_len:
        return np.concatenate((x, np.zeros(max_len - len(x))))
    else:
        return x[0 : max_len]
    
    
def calculate_scalar_of_tensor(x):
    if x.ndim == 2:
        axis = 0
    elif x.ndim == 3:
        axis = (0, 1)

    mean = np.mean(x, axis=axis)
    std = np.std(x, axis=axis)

    return mean, std


def load_scalar(scalar_path):
    with h5py.File(scalar_path, 'r') as hf:
        mean = hf['mean'][:]
        std = hf['std'][:]
        
    scalar = {'mean': mean, 'std': std}
    return scalar
    
    
def scale(x, mean, std):
    return (x - mean) / std
    
    
def inverse_scale(x, mean, std):
    return x * std + mean
    

def get_labels(taxonomy_level):
    if taxonomy_level == 'fine':
        return config.fine_labels
    elif taxonomy_level == 'coarse':
        return config.coarse_labels
    else:
        raise Exception('Incorrect argument!')
        
    return dict


def write_submission_csv(audio_names, outputs, taxonomy_level, submission_path):
    fine_labels = config.fine_labels
    coarse_labels = config.coarse_labels
    
    f = open(submission_path, 'w')
    
    head = ','.join(['audio_filename'] + fine_labels + coarse_labels)
    f.write('{}\n'.format(head))
    
    for n, audio_name in enumerate(audio_names):
        
        if taxonomy_level == 'fine':
            line = ','.join([audio_name] + list(map(str, outputs[n])) + ['0.'] * len(coarse_labels))
        elif taxonomy_level == 'coarse':
            line = ','.join([audio_name] + ['0.'] * len(fine_labels) + list(map(str, outputs[n])))
        else:
            raise Exception('Incorrect argument!')
            
        f.write('{}\n'.format(line))

    f.close()