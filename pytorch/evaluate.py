import os
import sys
sys.path.insert(1, os.path.join(sys.path[0], '../utils'))
sys.path.insert(1, os.path.join(sys.path[0], '../evaluation_tools'))

import numpy as np
import time
import logging
import matplotlib.pyplot as plt
from sklearn import metrics
import _pickle as cPickle
import sed_eval

from utilities import get_filename, inverse_scale, get_labels, write_submission_csv
from pytorch_utils import forward
import metrics as offical_metrics
import config


class Evaluator(object):
    def __init__(self, model, data_generator, taxonomy_level, statistics_path, cuda=True, 
        verbose=False):
        '''Evaluator to evaluate prediction performance. 
        
        Args: 
          model: object
          data_generator: object
          taxonomy_level: 'fine' | 'coarse'
          statistics_path: string
          cuda: bool
          verbose: bool
        '''

        self.model = model
        self.data_generator = data_generator
        self.taxonomy_level = taxonomy_level
        self.statistics_path = statistics_path
        self.cuda = cuda
        self.verbose = verbose
        
        self.frames_per_second = config.frames_per_second
        self.labels = get_labels(taxonomy_level)
        
        self.statistics = {
            'train': {'iteration': [], 'average_precision': [], 
                'micro_auprc': [], 'micro_f1': [], 'macro_auprc': []}, 
            'validate': {'iteration': [], 'average_precision': [], 
                'micro_auprc': [], 'micro_f1': [], 'macro_auprc': []}
            }

    def get_binary_target(self, target):
        '''Get binarized target. The original target is between 0 and 1
        representing the average annotations of labelers. Set a threshold to
        binarize the target to either 0 or 1. Pay attention this function is
        only used for validation. The evaluation data is manually verified and 
        the annotations are coherent. 
        '''
        
        threshold = 0.001   # XOR of annotations
        return (np.sign(target - threshold) + 1) / 2

    def evaluate(self, data_type, iteration, 
        submission_path=None, annotation_path=None, yaml_path=None, 
        max_iteration=None):
        '''Evaluate prediction performance. 
        
        Args:
          data_type: 'train' | 'validate'
          iteration: int
          submission_path: None | string
          annotation_path: None | string, path of ground truth csv
          yaml_path: None | string, path of yaml file
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
        
        average_precision = metrics.average_precision_score(target, output, average=None)
        
        if self.verbose:
            logging.info('{} average precision:'.format(data_type))        
            for k, label in enumerate(self.labels):
                logging.info('    {:<40}{:.3f}'.format(label, average_precision[k]))
            logging.info('    {:<40}{:.3f}'.format('Average', np.mean(average_precision)))
        else:
            logging.info('{}:'.format(data_type))
            logging.info('    mAP: {:.3f}'.format(np.mean(average_precision)))

        self.statistics[data_type]['iteration'].append(iteration)
        self.statistics[data_type]['average_precision'].append(average_precision)

        #TODO
        # Write submission and evaluate with official evaluation tool
        # https://github.com/sonyc-project/urban-sound-tagging-baseline
        if submission_path:
            write_submission_csv(
                audio_names=output_dict['audio_name'], 
                outputs=output, 
                taxonomy_level=self.taxonomy_level, 
                submission_path=submission_path)
                
            logging.info('    Write submission to {}'.format(submission_path))
            
            # The following code are from official evaluation code
            df_dict = offical_metrics.evaluate(
                prediction_path=submission_path,
                annotation_path=annotation_path,
                yaml_path=yaml_path,
                mode=self.taxonomy_level)
                            
            micro_auprc, eval_df = offical_metrics.micro_averaged_auprc(
                df_dict, return_df=True)
                
            macro_auprc, class_auprc = offical_metrics.macro_averaged_auprc(
                df_dict, return_classwise=True)
    
            # Get index of first threshold that is at least 0.5
            thresh_0pt5_idx = (eval_df['threshold'] >= 0.5).nonzero()[0][0]
    
            logging.info('    Official evaluation: ')
            logging.info('    Micro AUPRC:           {:.3f}'.format(micro_auprc))
            logging.info('    Micro F1-score (@0.5): {:.3f}'.format(eval_df['F'][thresh_0pt5_idx]))
            logging.info('    Macro AUPRC:           {:.3f}'.format(macro_auprc))
            
            self.statistics[data_type]['micro_auprc'].append(micro_auprc)
            self.statistics[data_type]['micro_f1'].append(eval_df['F'][thresh_0pt5_idx])
            self.statistics[data_type]['macro_auprc'].append(macro_auprc)
            
        if self.statistics_path:
            cPickle.dump(self.statistics, open(self.statistics_path, 'wb'))
            logging.info('    Dump statistics to {}'.format(self.statistics_path))
    
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
    