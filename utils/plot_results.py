import argparse
import os
import matplotlib.pyplot as plt
import _pickle as cPickle
import numpy as np

import config


def plot_results(args):
    # Arugments & parameters
    dataset_dir = args.dataset_dir
    workspace = args.workspace
    taxonomy_level = args.taxonomy_level
    
    filename = 'main'
    prefix = ''
    frames_per_second = config.frames_per_second
    mel_bins = config.mel_bins
    holdout_fold = 1
    max_plot_iteration = 5000
    data_type = 'validate'
    
    iterations = np.arange(0, max_plot_iteration, 200)
    
    def _load_stat(model_type):
        statistics_path = os.path.join(workspace, 'statistics', filename, 
            '{}logmel_{}frames_{}melbins'.format(prefix, frames_per_second, mel_bins), 
            model_type, 'taxonomy_level={}'.format(taxonomy_level), 
            'holdout_fold={}'.format(holdout_fold), 'validate_statistics.pickle')
        
        statistics_list = cPickle.load(open(statistics_path, 'rb'))
        average_precisions = np.array([statistics['average_precision'] for statistics in statistics_list])    # (N, classes_num)
        mAP = np.mean(average_precisions, axis=-1)
        micro_auprc = np.array([statistics['micro_auprc'] for statistics in statistics_list])
        micro_f1 = np.array([statistics['micro_f1'] for statistics in statistics_list])
        macro_auprc = np.array([statistics['macro_auprc'] for statistics in statistics_list])
        legend = '{}'.format(model_type)
        
        results = {'mAP': mAP, 'micro_auprc': micro_auprc, 'micro_f1': micro_f1, 'macro_auprc': macro_auprc, 'legend': legend}
        print('Model type: {}, mAP: {:.3f}, micro_auprc: {:.3f}, micro_f1: {:.3f}, macro_auprc: {:.3f}'.format(model_type, mAP[-1], micro_auprc[-1], micro_f1[-1], macro_auprc[-1]))
        
        return results
    
    fig, axs = plt.subplots(2, 2, figsize=(12, 8))
    
    
    keys = ['mAP', 'micro_auprc', 'micro_f1', 'macro_auprc']
        
    for n, key in enumerate(keys):
        lines = []
        
        results = _load_stat('Cnn_5layers_AvgPooling')
        line, = axs[n // 2, n % 2].plot(results[key], label=results['legend'])
        lines.append(line)

        results = _load_stat('Cnn_9layers_AvgPooling')
        line, = axs[n // 2, n % 2].plot(results[key], label=results['legend'])
        lines.append(line)
        
        results = _load_stat('Cnn_9layers_MaxPooling')
        line, = axs[n // 2, n % 2].plot(results[key], label=results['legend'])
        lines.append(line)

        results = _load_stat('Cnn_13layers_AvgPooling')
        line, = axs[n // 2, n % 2].plot(results[key], label=results['legend'])
        lines.append(line)
        
        axs[n // 2, n % 2].set_title(key)    
        axs[n // 2, n % 2].legend(handles=lines, loc=4)
        axs[n // 2, n % 2].set_ylim(0, 1.0)
        axs[n // 2, n % 2].grid(color='b', linestyle='solid', linewidth=0.2)
        axs[n // 2, n % 2].xaxis.set_ticks(np.arange(0, len(iterations), len(iterations) // 4))
        axs[n // 2, n % 2].xaxis.set_ticklabels(np.arange(0, max_plot_iteration, max_plot_iteration // 4))
        
    plt.tight_layout()
    fig_path = '_tmp.png'
    plt.savefig(fig_path)
    print('Save fig to {}'.format(fig_path))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--dataset_dir', type=str, required=True)
    parser.add_argument('--workspace', type=str, required=True)
    parser.add_argument('--taxonomy_level', type=str, choices=['fine', 'coarse'], required=True)

    args = parser.parse_args()
    
    plot_results(args)