import os
from collections import Counter
from math import ceil, floor

import matplotlib.pyplot as plt

from DataFormats.DbInstance import DbInstance
from run_experiments import read_json


# plots a histogram of the cbs score distributions of the given clusterings
def plot_histograms_clustering(input_file_par2, sbs_file, highlight_index, param_names, param_values_list,
                               label_list, max_cluster_amount=20, columns=2, bin_step=20, height=500, dpi=96,
                               output_file='', normalize=False, show_plot=False, max_id=10000000000):
    x_label = 'CBS (s)'
    y_label = 'Number of clusterings'

    data = []
    if isinstance(input_file_par2, list):
        for file in input_file_par2:
            data = data + read_json(file)
    else:
        data = read_json(input_file_par2)

    rows = ceil(len(param_values_list[highlight_index]) / columns)

    min_b = 5000
    max_b = 0

    split_data = {}
    for value in param_values_list[highlight_index]:
        split_data[str(value)] = []

    for evaluation in data:
        if evaluation['id'] > max_id:
            break

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

    sbs = read_json(sbs_file)
    n_bins = [sbs[0]]
    while n_bins[-1] > min_b - bin_step:
        n_bins.append(n_bins[-1] - bin_step)
    n_bins.reverse()

    # n_bins = range(min_b - bin_step, max_b + bin_step, bin_step)
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

        if sbs_file != '':
            sbs_data = read_json(sbs_file)
            axes_flat[idx].axvline(x=sbs_data[0], color='r', linestyle='--')

    for idx in range(len(param_values_list[highlight_index]), len(axes_flat)):
        axes_flat[idx].set_visible(False)

    if output_file != '':
        plt.savefig(os.environ['TEXPATH'] + output_file + '.svg')

    if show_plot:
        plt.show()


# plots a boxplot of the cbs score distributions of the given clusterings
def plot_boxplot_clustering(input_file_par2, highlight_index, param_names, param_values_list,
                            label_list, max_cluster_amount=20, dpi=96, angle=90, y_axis_range=None,
                            output_file='', show_plot=False, sbs_file='', remove_box_if_all_values_in_range_of_sbs=None, max_id=100000000):
    data = []
    if isinstance(input_file_par2, list):
        for file in input_file_par2:
            data = data + read_json(file)
    else:
        data = read_json(input_file_par2)

    split_data = {}
    for value in param_values_list[highlight_index]:
        split_data[str(value)] = []

    for evaluation in data:
        if evaluation['id'] > max_id:
            break

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
    plot_labels = []
    for label, value in zip(label_list, param_values_list[highlight_index]):
        if remove_box_if_all_values_in_range_of_sbs is not None and sbs_file != '':
            sbs_data = read_json(sbs_file)
            found_bigger = False
            for score in split_data[str(value)]:
                if score < (sbs_data[0] - remove_box_if_all_values_in_range_of_sbs):
                    found_bigger = True
                    break

            if found_bigger:
                plot_data.append(split_data[str(value)])
                plot_labels.append(label)
        else:
            plot_data.append(split_data[str(value)])
            plot_labels.append(label)

    fig = plt.figure(figsize=(1700 / dpi, 1000 / dpi), dpi=dpi,
                     constrained_layout=True)

    ax = fig.add_subplot(111)

    ax.boxplot(plot_data)
    plt.xticks(range(1, len(plot_labels) + 1), plot_labels, rotation=angle)
    ax.set_ylabel('CBS (s)')

    if sbs_file != '':
        sbs_data = read_json(sbs_file)
        plt.axhline(y=sbs_data[0], color='r', linestyle='--')

    if y_axis_range is not None:
        ax = plt.gca()
        ax.set_ylim(y_axis_range)

    if output_file != '':
        fig.savefig(os.environ['TEXPATH'] + output_file + '.svg')

    if show_plot:
        fig.show()


# plots a boxplot of family distribution
def plot_boxplot_family(db_instance: DbInstance, dpi=96, output_file='', show_plot=False):
    family_list = [family[0] for family in db_instance.family_wh]
    family_count = Counter(family_list)
    count_list = [value for key, value in family_count.items()]

    fig = plt.figure(figsize=(1700 / dpi, 500 / dpi), dpi=dpi,
                     constrained_layout=True)

    ax = fig.add_subplot(111)
    ax.set_xlabel('Number of family instances')

    ax.boxplot(count_list, vert=False, widths=[1.7])

    if output_file != '':
        fig.savefig(os.environ['TEXPATH'] + output_file + '.svg')

    if show_plot:
        fig.show()
