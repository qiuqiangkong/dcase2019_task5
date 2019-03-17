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
    def __init__(self, model, data_generator, taxonomy_level, cuda=True, 
        verbose=False):
        '''Evaluator to evaluate prediction performance. 
        
        Args: 
          model: object
          data_generator: object
          taxonomy_level: 'fine' | 'coarse'
          cuda: bool
          verbose: bool
        '''

        self.model = model
        self.data_generator = data_generator
        self.taxonomy_level = taxonomy_level
        self.cuda = cuda
        self.verbose = verbose
        
        self.frames_per_second = config.frames_per_second
        self.labels = get_labels(taxonomy_level)

    def get_binary_target(self, target):
        '''Get binarized target. The original target is between 0 and 1
        representing the average annotations of labelers. Set a threshold to
        binarize the target to either 0 or 1. Pay attention this function is
        only used for validation. The evaluation data is manually verified and 
        the annotations are coherent. 
        '''
        
        threshold = 0.099   # If at least one labeler labels the sound class
                            # to be presence, then the target is set to 1. 
        return (np.sign(target - threshold) + 1) / 2

    def evaluate(self, data_type, submission_path, max_iteration=None):
        '''Evaluate prediction performance. 
        
        Args:
          data_type: 'train' | 'validate'
          submission_path: None | string
          max_iteration: None | int, use maximum iteration of partial data for
              fast evaluation
        '''
        
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

        #TODO
        # Write submission and evaluate with official evaluation tool
    
    def visualize(self, data_type, max_iteration=None):
        '''Visualize the log mel spectrogram. 
        
        Args:
          data_type: 'train' | 'validate'
          max_iteration: None | int, use maximum iteration of partial data for
              fast evaluation
        '''

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
                    row = k // cols_num
                    col = k % cols_num
                    title = coarse_idx_to_lb[k]
                    title = '{}\n{}'.format(coarse_idx_to_lb[k], audio_name)
                    axs[row, col].set_title(title, color='r')
                    logmel = inverse_scale(output_dict['feature'][n], 
                        self.data_generator.scalar['mean'], 
                        self.data_generator.scalar['std'])
                    axs[row, col].matshow(logmel.T, origin='lower', aspect='auto', cmap='jet')                
                    axs[row, col].set_xticks([0, frames_num])
                    axs[row, col].set_xticklabels(['0', '{:.1f} s'.format(audio_duration)])
                    axs[row, col].xaxis.set_ticks_position('bottom')
                    axs[row, col].set_ylabel('Mel bins')
                    axs[row, col].set_yticks([])
                    break
        
        for k in range(coarse_classes_num, rows_num * cols_num):
            row = k // cols_num
            col = k % cols_num
            axs[row, col].set_visible(False)
            
        fig.tight_layout(pad=0, w_pad=0, h_pad=0)
        plt.show()
    