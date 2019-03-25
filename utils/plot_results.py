import argparse
import os
import matplotlib.pyplot as plt
import _pickle as cPickle
import numpy as np


def plot_results(args):
    # Arugments & parameters
    dataset_dir = args.dataset_dir
    workspace = args.workspace
    taxonomy_level = args.taxonomy_level
    select = args.select
    
    filename = 'main'
    prefix = ''
    frames_per_second = 64
    mel_bins = 64
    holdout_fold = 1
    max_plot_iteration = 5000
    data_type = 'validate'
    
    iterations = np.arange(0, max_plot_iteration, 200)
    
    def _load_stat(model_type):
        statistics_path = os.path.join(workspace, 'statistics', filename, 
            '{}logmel_{}frames_{}melbins'.format(prefix, frames_per_second, mel_bins), 
            model_type, 'taxonomy_level={}'.format(taxonomy_level), 
            'holdout_fold={}'.format(holdout_fold), 'statistics.pickle')
        
        statistics = cPickle.load(open(statistics_path, 'rb'))
        average_precision_matrix = np.array(statistics[data_type]['average_precision'])    # (N, classes_num)
        mAP = np.mean(average_precision_matrix, axis=-1)
        micro_auprc = np.array(statistics[data_type]['micro_auprc'])
        micro_f1 = np.array(statistics[data_type]['micro_f1'])
        macro_auprc = np.array(statistics[data_type]['macro_auprc'])
        legend = '{}'.format(model_type)
        return mAP, micro_auprc, micro_f1, macro_auprc, legend
    
    fig, axs = plt.subplots(2, 2, figsize=(12, 8))
    
    
    titles = ['mAP', 'micro AUPRC', 'micro F1', 'macro AUPRC']
        
    for n in range(4):
        lines = []
        
        results = _load_stat('Cnn_9layers_MaxPooling')
        line, = axs[n // 2, n % 2].plot(results[n], label=results[-1])
        lines.append(line)
        
        results = _load_stat('Cnn_9layers_AvgPooling')
        line, = axs[n // 2, n % 2].plot(results[n], label=results[-1])
        lines.append(line)

        results = _load_stat('Cnn_13layers_AvgPooling')
        line, = axs[n // 2, n % 2].plot(results[n], label=results[-1])
        lines.append(line)
        
        axs[n // 2, n % 2].set_title(titles[n])    
        axs[n // 2, n % 2].legend(handles=lines, loc=4)
        axs[n // 2, n % 2].set_ylim(0, 1.0)
        axs[n // 2, n % 2].grid(color='b', linestyle='solid', linewidth=0.2)
        axs[n // 2, n % 2].xaxis.set_ticks(np.arange(0, len(iterations), len(iterations) // 4))
        axs[n // 2, n % 2].xaxis.set_ticklabels(np.arange(0, max_plot_iteration, max_plot_iteration // 4))
        
    plt.tight_layout()
    # plt.show()
    plt.savefig('_tmp.png')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--dataset_dir', type=str, required=True)
    parser.add_argument('--workspace', type=str, required=True)
    parser.add_argument('--taxonomy_level', type=str, choices=['fine', 'coarse'], required=True)
    parser.add_argument('--select', type=int, required=True)

    args = parser.parse_args()
    
    plot_results(args)