import os
import sys
sys.path.insert(1, os.path.join(sys.path[0], 'utils'))
import numpy as np
import argparse
import h5py
import librosa
from scipy import signal
import matplotlib.pyplot as plt
import time
import math
import pandas as pd
import random

from utilities import (create_folder, read_audio, calculate_scalar_of_tensor, 
    pad_truncate_sequence)
import config


class LogMelExtractor(object):
    def __init__(self, sample_rate, window_size, hop_size, mel_bins, fmin, fmax):
        '''Log mel feature extractor. 
        
        Args:
          sample_rate: int
          window_size: int
          hop_size: int
          mel_bins: int
          fmin: int, minimum frequency of mel filter banks
          fmax: int, maximum frequency of mel filter banks
        '''
        
        self.window_size = window_size
        self.hop_size = hop_size
        self.window_func = np.hanning(window_size)
        
        self.melW = librosa.filters.mel(
            sr=sample_rate, 
            n_fft=window_size, 
            n_mels=mel_bins, 
            fmin=fmin, 
            fmax=fmax).T
        '''(n_fft // 2 + 1, mel_bins)'''

    def transform(self, audio):
        '''Extract feature of a singlechannel audio file. 
        
        Args:
          audio: (samples,)
          
        Returns:
          feature: (frames_num, freq_bins)
        '''
    
        window_size = self.window_size
        hop_size = self.hop_size
        window_func = self.window_func
        
        # Compute short-time Fourier transform
        stft_matrix = librosa.core.stft(
            y=audio, 
            n_fft=window_size, 
            hop_length=hop_size, 
            window=window_func, 
            center=True, 
            dtype=np.complex64, 
            pad_mode='reflect').T
        '''(N, n_fft // 2 + 1)'''
    
        # Mel spectrogram
        mel_spectrogram = np.dot(np.abs(stft_matrix) ** 2, self.melW)
        
        # Log mel spectrogram
        logmel_spectrogram = librosa.core.power_to_db(
            mel_spectrogram, ref=1.0, amin=1e-10, 
            top_db=None)
        
        logmel_spectrogram = logmel_spectrogram.astype(np.float32)
        
        return logmel_spectrogram


def read_metadata(metadata_path, data_type, mini_data):
    '''Read metadata csv file. 
    
    Args:
      metadata_path: string
      data_type: 'train' | 'validate'
      mini_data: bool, set True for debugging on a small part of data
      
    Returns:
      meta_dict
    '''

    fine_labels = config.fine_labels
    fine_classes_num = config.fine_classes_num
    fine_lb_to_idx = config.fine_lb_to_idx
    coarse_labels = config.coarse_labels
    coarse_classes_num = config.coarse_classes_num
    coarse_lb_to_idx = config.coarse_lb_to_idx

    # Read csv data
    assert data_type in ['train', 'validate']
    df = pd.read_csv(metadata_path, sep=',')
    df = df[df['split'] == data_type].reset_index()
    
    # Each audio may be annotated by multiple labelers. So remove duplicate 
    # audio names. 
    audio_names = np.array(sorted(list(set(df['audio_filename']))))
    
    if mini_data:
        random_state = np.random.RandomState(1234)
        random_state.shuffle(audio_names)
        audio_names = audio_names[0 : 10]

    fine_targets = []
    coarse_targets = []
    
    for audio_name in audio_names:
        # One audio_name may have multiple labelers
        indexes = df.index[df['audio_filename'] == audio_name]
        fine_target = np.zeros(fine_classes_num)
        coarse_target = np.zeros(coarse_classes_num)
        
        # Aggregating annotation of multiple labelers
        for index in indexes:
            for fine_label in fine_labels:
                class_idx = fine_lb_to_idx[fine_label]
                label_type = '{}_presence'.format(fine_label)
                fine_target[class_idx] += df.iloc[index][label_type]
            
            for coarse_label in coarse_labels:
                class_idx = coarse_lb_to_idx[coarse_label]
                label_type = '{}_presence'.format(coarse_label)
                coarse_target[class_idx] += df.iloc[index][label_type]
                
        # Annotation of an audio is the average annotation of multiple labelrs
        fine_target /= len(indexes)
        coarse_target /= len(indexes)
        
        fine_targets.append(fine_target)
        coarse_targets.append(coarse_target)
        
    fine_targets = np.array(fine_targets)
    coarse_targets = np.array(coarse_targets)
    
    meta_dict = {
        'audio_name': audio_names, 
        'fine_target': fine_targets, 
        'coarse_target': coarse_targets}

    return meta_dict


def read_evaluate_metadata(audios_dir, mini_data):
    audio_names = sorted(os.listdir(audios_dir))

    if mini_data:
        audio_names = audio_names[0 : 10]

    meta_dict = {'audio_name': audio_names}
    
    return meta_dict


def calculate_feature_for_all_audio_files(args):
    '''Calculate feature of audio files and write out features to a single hdf5 
    file. 
    
    Args:
      dataset_dir: string
      workspace: string
      data_type: 'train' | 'validate' | 'evaluate'
      mini_data: bool, set True for debugging on a small part of data
    '''
    
    # Arguments & parameters
    dataset_dir = args.dataset_dir
    data_type = args.data_type
    workspace = args.workspace
    mini_data = args.mini_data
    
    sample_rate = config.sample_rate
    window_size = config.window_size
    hop_size = config.hop_size
    mel_bins = config.mel_bins
    fmin = config.fmin
    fmax = config.fmax
    frames_per_second = config.frames_per_second
    frames_num = config.frames_num
    total_samples = config.total_samples
    
    # Paths
    if mini_data:
        prefix = 'minidata_'
    else:
        prefix = ''
        
    metadata_path = os.path.join(dataset_dir, 'annotations.csv')

    if data_type in ['train', 'validate']:
        audios_dir = os.path.join(dataset_dir, data_type)
    elif data_type == 'evaluate':
        audios_dir = os.path.join(dataset_dir, 'audio-eval')
    
    feature_path = os.path.join(workspace, 'features', 
        '{}logmel_{}frames_{}melbins'.format(prefix, frames_per_second, mel_bins), 
        '{}.h5'.format(data_type))
    create_folder(os.path.dirname(feature_path))
        
    # Feature extractor
    feature_extractor = LogMelExtractor(
        sample_rate=sample_rate, 
        window_size=window_size, 
        hop_size=hop_size, 
        mel_bins=mel_bins, 
        fmin=fmin, 
        fmax=fmax)

    # Read metadata
    print('Extracting features of all audio files ...')
    extract_time = time.time()
    
    if data_type in ['train', 'validate']:
        meta_dict = read_metadata(metadata_path, data_type, mini_data)
    elif data_type == 'evaluate':
        meta_dict = read_evaluate_metadata(audios_dir, mini_data)

    # Hdf5 containing features and targets
    hf = h5py.File(feature_path, 'w')

    hf.create_dataset(
        name='audio_name', 
        data=[audio_name.encode() for audio_name in meta_dict['audio_name']], 
        dtype='S32')

    if 'fine_target' in meta_dict.keys():
        hf.create_dataset(
            name='fine_target', 
            data=meta_dict['fine_target'], 
            dtype=np.float32)
            
    if 'coarse_target' in meta_dict.keys():
        hf.create_dataset(
            name='coarse_target', 
            data=meta_dict['coarse_target'], 
            dtype=np.float32)

    hf.create_dataset(
        name='feature', 
        shape=(0, frames_num, mel_bins), 
        maxshape=(None, frames_num, mel_bins), 
        dtype=np.float32)

    for (n, audio_name) in enumerate(meta_dict['audio_name']):
        audio_path = os.path.join(audios_dir, audio_name)
        print(n, audio_path)
        
        # Read audio
        (audio, _) = read_audio(
            audio_path=audio_path, 
            target_fs=sample_rate)

        # Pad or truncate audio recording
        audio = pad_truncate_sequence(audio, total_samples)
        
        # Extract feature
        feature = feature_extractor.transform(audio)
        
        # Remove the extra frames caused by padding zero
        feature = feature[0 : frames_num]
        
        hf['feature'].resize((n + 1, frames_num, mel_bins))
        hf['feature'][n] = feature
        
    hf.close()
        
    print('Write hdf5 file to {} using {:.3f} s'.format(
        feature_path, time.time() - extract_time))
    
    
def calculate_scalar(args):
    '''Calculate and write out scalar of features. 
    
    Args:
      workspace: string
      data_type: 'train'
      mini_data: bool, set True for debugging on a small part of data
    '''

    # Arguments & parameters
    data_type = args.data_type
    workspace = args.workspace
    mini_data = args.mini_data
    
    mel_bins = config.mel_bins
    frames_per_second = config.frames_per_second
    
    # Paths
    if mini_data:
        prefix = 'minidata_'
    else:
        prefix = ''
    
    feature_path = os.path.join(workspace, 'features', 
        '{}logmel_{}frames_{}melbins'.format(prefix, frames_per_second, mel_bins), 
        '{}.h5'.format(data_type))
        
    scalar_path = os.path.join(workspace, 'scalars', 
        '{}logmel_{}frames_{}melbins'.format(prefix, frames_per_second, mel_bins), 
        '{}.h5'.format(data_type))
    create_folder(os.path.dirname(scalar_path))
        
    # Load data
    load_time = time.time()
    
    with h5py.File(feature_path, 'r') as hf:
        features = hf['feature'][:]
    
    # Calculate scalar
    features = np.concatenate(features, axis=0)
    (mean, std) = calculate_scalar_of_tensor(features)
    
    with h5py.File(scalar_path, 'w') as hf:
        hf.create_dataset('mean', data=mean, dtype=np.float32)
        hf.create_dataset('std', data=std, dtype=np.float32)
    
    print('All features: {}'.format(features.shape))
    print('mean: {}'.format(mean))
    print('std: {}'.format(std))
    print('Write out scalar to {}'.format(scalar_path))
            

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='')
    subparsers = parser.add_subparsers(dest='mode')

    # Calculate feature for all audio files
    parser_logmel = subparsers.add_parser('calculate_feature_for_all_audio_files')
    parser_logmel.add_argument('--dataset_dir', type=str, required=True, help='Directory of dataset.')
    parser_logmel.add_argument('--workspace', type=str, required=True, help='Directory of your workspace.')
    parser_logmel.add_argument('--data_type', type=str, choices=['train', 'validate', 'evaluate'], required=True)
    parser_logmel.add_argument('--mini_data', action='store_true', default=False, help='Set True for debugging on a small part of data.')

    # Calculate scalar
    parser_scalar = subparsers.add_parser('calculate_scalar')
    parser_scalar.add_argument('--data_type', type=str, choices=['train'], required=True, help='Scalar is calculated on train data.')
    parser_scalar.add_argument('--workspace', type=str, required=True, help='Directory of your workspace.')
    parser_scalar.add_argument('--mini_data', action='store_true', default=False, help='Set True for debugging on a small part of data.')
    
    # Parse arguments
    args = parser.parse_args()
    
    if args.mode == 'calculate_feature_for_all_audio_files':
        calculate_feature_for_all_audio_files(args)
        
    elif args.mode == 'calculate_scalar':
        calculate_scalar(args)
        
    else:
        raise Exception('Incorrect arguments!')