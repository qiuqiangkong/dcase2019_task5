import os
import sys
sys.path.insert(1, os.path.join(sys.path[0], '../utils'))

import numpy as np
import time
import logging
import matplotlib.pyplot as plt
from sklearn import metrics
import sed_eval

from utilities import get_filename, inverse_scale, get_labels
from pytorch_utils import forward
import config


class Evaluator(object):
    def __init__(self, model, data_generator, taxonomy_level, cuda=True, verbose=False):

        self.model = model
        self.data_generator = data_generator
        self.taxonomy_level = taxonomy_level
        self.cuda = cuda
        self.verbose = verbose
        
        self.frames_per_second = config.frames_per_second
        self.labels = get_labels(taxonomy_level)

    def get_binary_target(self, target):
        threshold = 0.099
        return (np.sign(target - threshold) + 1) / 2

    def evaluate(self, data_type, submission_path, max_iteration=None):

        generate_func = self.data_generator.generate_validate(
            data_type=data_type, 
            max_iteration=max_iteration)
        
        # Forward
        output_dict = forward(
            model=self.model, 
            generate_func=generate_func, 
            cuda=self.cuda, 
            return_target=True)
            
        output = output_dict['output']
        target = output_dict['{}_target'.format(self.taxonomy_level)]
        target = self.get_binary_target(target)
        
        mAP = metrics.average_precision_score(target, output, average=None)
        
        
        if self.verbose:
            logging.info('{} average precision:'.format(data_type))        
            for k, label in enumerate(self.labels):
                logging.info('    {:<40}{:.3f}'.format(label, mAP[k]))
            logging.info('    {:<40}{:.3f}'.format('Avg.', np.mean(mAP)))
        else:
            logging.info('{} mAP: {:.3f}'.format(data_type, np.mean(mAP)))
    
    def visualize(self, data_type, max_iteration=None):

        mel_bins = config.mel_bins
        audio_duration = config.audio_duration
        frames_num = config.frames_num
        coarse_classes_num = config.coarse_classes_num
        coarse_idx_to_lb = config.coarse_idx_to_lb
        
        generate_func = self.data_generator.generate_validate(
            data_type=data_type, 
            max_iteration=max_iteration)
        
        # Forward
        output_dict = forward(
            model=self.model, 
            generate_func=generate_func, 
            cuda=self.cuda, 
            return_input=True, 
            return_target=True)

        rows_num = 3
        cols_num = 3
        
        fig, axs = plt.subplots(rows_num, cols_num, figsize=(10, 5))

        for k in range(coarse_classes_num):
            for n, audio_name in enumerate(output_dict['audio_name']):
                if output_dict['coarse_target'][n, k] > 0.5:
                    title = coarse_idx_to_lb[k]
                    title = '{}\n{}'.format(coarse_idx_to_lb[k], audio_name)
                    axs[k // cols_num, k % cols_num].set_title(title, color='r')
                    logmel = inverse_scale(output_dict['feature'][n], self.data_generator.scalar['mean'], self.data_generator.scalar['std'])
                    axs[k // cols_num, k % cols_num].matshow(logmel.T, origin='lower', aspect='auto', cmap='jet')                
                    axs[k // cols_num, k % cols_num].set_xticks([0, frames_num])
                    axs[k // cols_num, k % cols_num].set_xticklabels(['0', '{:.1f} s'.format(audio_duration)])
                    axs[k // cols_num, k % cols_num].xaxis.set_ticks_position('bottom')
                    axs[k // cols_num, k % cols_num].set_ylabel('Mel bins')
                    axs[k // cols_num, k % cols_num].set_yticks([])
                    break
        
        for k in range(coarse_classes_num, rows_num * cols_num):
            row = k // cols_num
            col = k % cols_num
            axs[row, col].set_visible(False)
            
        fig.tight_layout(pad=0, w_pad=0, h_pad=0)
        plt.show()
    