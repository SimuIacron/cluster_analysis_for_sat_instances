import os
from math import ceil, floor

import matplotlib.pyplot as plt

from run_experiments import read_json


def plot_histograms_clustering(input_file_par2, highlight_index, param_names, param_values_list,
                               label_list, max_cluster_amount=20, columns=2, bin_step=20, height=500, dpi=96,
                               output_file='', normalize=False, show_plot=False):
    x_label = 'CBS (s)'
    y_label = 'frequency'

    data = read_json(input_file_par2)

    rows = ceil(len(param_values_list[highlight_index]) / columns)

    min_b = 5000
    max_b = 0

    split_data = {}
    for value in param_values_list[highlight_index]:
        split_data[str(value)] = []

    for evaluation in data:
        eval_is_in_graph = True
        for idx, value in enumerate(param_names):
            if evaluation['settings'][value] not in param_values_list[idx]:
                eval_is_in_graph = False
                break

        if eval_is_in_graph and len(evaluation['clusters']) <= max_cluster_amount:

            if evaluation['par2'][0] < min_b:
                min_b = floor(evaluation['par2'][0])
            if evaluation['par2'][0] > max_b:
                max_b = ceil(evaluation['par2'][0])

            for value in param_values_list[highlight_index]:
                if str(evaluation['settings'][param_names[highlight_index]]) == str(value):
                    split_data[str(value)].append(evaluation['par2'][0])
                    break

    n_bins = range(min_b - bin_step, max_b + bin_step, bin_step)
    fig, axes = plt.subplots(nrows=rows, ncols=columns, figsize=(1700 / dpi, 1000 / dpi), dpi=dpi,
                             constrained_layout=True)
    axes_flat = [axes]
    if len(param_values_list[highlight_index]) > 1:
        axes_flat = axes.flat

    for idx, value in enumerate(param_values_list[highlight_index]):
        axes_flat[idx].set_title(label_list[idx])
        axes_flat[idx].set_ylim([0, height])
        axes_flat[idx].set_xlabel(x_label)
        axes_flat[idx].set_ylabel(y_label)
        axes_flat[idx].hist(split_data[str(value)], n_bins, density=normalize)

    for idx in range(len(param_values_list[highlight_index]), len(axes_flat)):
        axes_flat[idx].set_visible(False)

    if output_file != '':
        plt.savefig(os.environ['TEXPATH'] + output_file + '.svg')

    if show_plot:
        plt.show()


def plot_boxplot_clustering(input_file_par2, highlight_index, param_names, param_values_list,
                            label_list, max_cluster_amount=20, dpi=96, angle=90, y_axis_range=None,
                            output_file='', show_plot=False):
    data = read_json(input_file_par2)
    split_data = {}
    for value in param_values_list[highlight_index]:
        split_data[str(value)] = []

    for evaluation in data:
        eval_is_in_graph = True
        for idx, value in enumerate(param_names):
            if evaluation['settings'][value] not in param_values_list[idx]:
                eval_is_in_graph = False
                break

        if eval_is_in_graph and len(evaluation['clusters']) <= max_cluster_amount:
            for value in param_values_list[highlight_index]:
                if str(evaluation['settings'][param_names[highlight_index]]) == str(value):
                    split_data[str(value)].append(evaluation['par2'][0])
                    break

    plot_data = []
    for value in param_values_list[highlight_index]:
        plot_data.append(split_data[str(value)])

    fig = plt.figure(figsize=(1700 / dpi, 1000 / dpi), dpi=dpi,
                     constrained_layout=True)

    ax = fig.add_subplot(111)

    ax.boxplot(plot_data)
    plt.xticks(range(1, len(label_list) + 1), label_list, rotation=angle)

    if y_axis_range is not None:
        ax = plt.gca()
        ax.set_ylim(y_axis_range)

    if output_file != '':
        fig.savefig(os.environ['TEXPATH'] + output_file + '.svg')

    if show_plot:
        fig.show()
