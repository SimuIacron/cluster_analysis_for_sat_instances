import itertools
import os
from collections import Counter
from math import ceil

import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from numpy import argmin
import plotly.express as px

from util_scripts import util, exportFigure
from DataFormats.DbInstance import DbInstance
from run_experiments import read_json, write_json


# gets the evaluations (or experiments) depending on the given settings_dict from the input_file
# the settings dict, needs to have the same structure as the dict stored in "settings" in the input file,
# however instead of values, for each settings list of settings is used, to enable the usage of "or"
# example: In the file the structure of the problem is given as {settings: {"scaling_algorithm": "SCALEMINUSPLUS1" }}
# then the settings dict needs to hve the structure: {"scaling_algorithm": ["SCALEMINUSPLUS1"] }, we can especially use
# {"scaling_algorithm": ["SCALEMINUSPLUS1", "NORMALSCALE"] } to select evaluations with different scaling algorithm

# returns the list of evaluations that fit the dict and a diff which contains all keys, that where missing in the
# settings_dict, but where present in at least one selected evaluation
def collect_evaluation(input_file, settings_dict):
    data = read_json(input_file)
    collected_evaluation = []
    diff_set = set()
    for evaluation in data:
        matchingValues = True
        for key, value in settings_dict.items():
            if evaluation['settings'][key] not in value:
                matchingValues = False
                break

        if matchingValues:
            collected_evaluation.append(evaluation)
            diff = evaluation['settings'].keys() - settings_dict.keys()
            diff_set = set.union(diff_set, diff)

    return collected_evaluation, diff_set


# ----------------------------------------------------------------------------------------------------------------------

# Plots a circle diagram of the distribution of the virtual best solver
# input_file_bss: The file to read the virtual best solver data from
# output_file: The filename of the exported html (no export if equal to '')
# show_plot: If the plot should be opened in the browser after running the function
def plot_par2_vbs_distribution(input_file_vbs, output_file='', show_plot=False, ):
    result_bss = read_json(input_file_vbs)
    best_solver = []
    for key, item in result_bss[1].items():
        best_solver.append(item[0][0][0])

    counter = Counter(best_solver)

    keys = []
    values = []
    for key, value in counter.items():
        keys.append(key)
        values.append(value)

    fig = go.Figure(data=[go.Pie(labels=keys,
                                 values=values)],
                    layout=go.Layout(
                        title=go.layout.Title(
                            text='Distributions of solvers if each instance uses the virtual best solver'))
                    )

    if output_file != '':
        exportFigure.export_plot_as_html(fig, output_file)

    if show_plot:
        fig.show()


# Example

# plot_par2_vbs_distribution('vbs_without_glucose_syrup_yalsat', 'vbs_without_glucose_syrup_yalsat', show_plot=True)

# ----------------------------------------------------------------------------------------------------------------------

# Creates a bar chart of the solvers with the best mean par2 scores and for comparison the best single solver (bss)
# input_files_par2_scores: The file to read the par2 scores from
# input_file_vbs: The file to read the vbs score from
# input_file_sbs: The file to read the bss scores from
# plot_description: The title/description of the plot
# param_name: The name of the parameter to compare to in the settings of the instances
# value_list: The values of the parameter param_name that should be compared
# label_list: The names of the values of the parameter that should be shown as the labels of the legend
# helpful, if the values of the parameter are not nice for presentation
# max_cluster_amount: The maximum amount of clusters an instance in the plot should have
# cutoff: How many instance should be shown in the plot
# output_file: The filename of the exported html (no export if equal to '')
# show_plot: If the plot should be opened in the browser after running the function
# use_mat_plot: Whether to show the plot as a matplotlib plot or a plotly plot
def plot_cpar2_comparison(input_files_par2_scores, input_file_vbs, input_file_sbs, plot_description, highlight_index,
                          param_names,
                          value_lists, label_list, max_cluster_amount, cutoff, output_file='', show_plot=False,
                          use_mat_plot=True, use_dash_plot=False, show_complete_legend=False):
    x_label = 'Best Instances sorted by CPar2'
    y_label = 'CPar2 Score (s)'
    vbs_label = 'Virtual Best Solver'
    sbs_label = 'Single Best Solver'

    data = []
    for input_file in input_files_par2_scores:
        data = data + read_json(input_file)
    bss_data = read_json(input_file_sbs)
    vbs_data = read_json(input_file_vbs)

    sorted_data = sorted(data, key=lambda d: d['par2'][0])
    keys = [0]

    value_dict = {}
    for elem in value_lists[highlight_index]:
        value_dict[str(elem)] = [0]

    vbs = [vbs_data[0]]
    bss = [0]
    text = [vbs_label]
    counter = 1
    for evaluation in sorted_data:
        if len(keys) >= cutoff:
            break

        eval_is_in_graph = True
        for idx, value in enumerate(param_names):
            if evaluation['settings'][value] not in value_lists[idx]:
                eval_is_in_graph = False
                break

        if eval_is_in_graph and len(evaluation['clusters']) <= max_cluster_amount:

            for key, value in value_dict.items():
                if str(key) == str(evaluation['settings'][param_names[highlight_index]]):
                    value_dict[str(key)].append(evaluation['par2'][0])
                else:
                    value_dict[str(key)].append(0)

            bss.append(0)
            vbs.append(0)
            keys.append(counter)
            counter = counter + 1

            text_string = util.add_line_breaks_to_text(str(evaluation['settings']), ',', 5)

            text.append(text_string + ' cluster count: ' + str(len(evaluation['clusters'])))

    keys.append(counter)
    for elem in value_lists[highlight_index]:
        value_dict[str(elem)].append(0)
    bss.append(bss_data[0])
    vbs.append(0)
    text.append(sbs_label)

    if use_mat_plot:

        barwidth = 1.2

        dpi = 150
        plt.figure(figsize=(1200 / dpi, 600 / dpi), dpi=dpi)
        plt.bar(keys, vbs, label=vbs_label, width=barwidth)
        plt.bar(keys, bss, label=sbs_label, width=barwidth)
        for idx, elem in enumerate(value_lists[highlight_index]):

            has_value_not_zero = False
            for item in value_dict[str(elem)]:
                if item != 0:
                    has_value_not_zero = True
                    break

            if has_value_not_zero or show_complete_legend:
                plt.bar(keys, value_dict[str(elem)], label=label_list[idx], width=barwidth)

        plt.xlabel(x_label)
        plt.ylabel(y_label)
        ax = plt.subplot(111)
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.title(plot_description)

        if output_file != '':
            plt.savefig(os.environ['EXPPATH'] + output_file + '.svg')
        if show_plot:
            plt.show()

    if use_dash_plot:
        chart_data = [
            go.Bar(name=vbs_label, x=keys, y=vbs, hovertext=text),
            go.Bar(name=sbs_label, x=keys, y=bss, hovertext=text)
        ]

        for idx, key in enumerate(value_lists[highlight_index]):

            has_value_not_zero = False
            for item in value_dict[str(key)]:
                if item != 0:
                    has_value_not_zero = True
                    break

            if has_value_not_zero or show_complete_legend:
                chart_data.append(
                    go.Bar(name=label_list[idx], x=keys, y=value_dict[str(key)], hovertext=text)
                )

        fig = go.Figure(layout=go.Layout(
            title=go.layout.Title(text=plot_description)
        ),
            data=chart_data)

        fig.update_layout(barmode='stack', bargap=0)
        fig.update_xaxes(title_text=x_label)
        fig.update_yaxes(title_text=y_label)

        if output_file != '':
            exportFigure.export_plot_as_html(fig, output_file)
        if show_plot:
            fig.show()


# temp_solver_features = DatabaseReader.FEATURES_SOLVER.copy()
# temp_solver_features.pop(14)
# temp_solver_features.pop(7)
# input_dbs = [DatabaseReader.FEATURES_BASE, DatabaseReader.FEATURES_GATE, temp_solver_features]
# output = sum([list(map(list, itertools.combinations(input_dbs, i))) for i in range(len(input_dbs) + 1)], [])
# output_merged = []
# for combination in output:
#     comb = []
#     for elem in combination:
#         comb = comb + elem
#     output_merged.append(comb)
#
# plot_cpar2_comparison(['clustering_general/clustering_general_par2'], 'vbs_sbs/vbs', 'vbs_sbs/sbs',
#                       'CPar2 scores of different cluster algorithms using combinations of base, gate, runtimes',
#                       0, ['selected_data'],
#                       [output_merged[1:]],
#                       ['base', 'gate', 'runtimes', 'base gate', 'base runtimes', 'gate runtimes', 'base gate runtimes'],
#                       20, 200, output_file='clustering_general_plot_comb_base_gate_runtimes_best_cluster',
#                       show_plot=False,
#                       use_mat_plot=True, use_dash_plot=True)

# plot_cpar2_comparison(['clustering_general/clustering_general_par2'], 'vbs_sbs/vbs', 'vbs_sbs/sbs',
#                       'CPar2 scores of different cluster algorithms using combinations of base, gate, runtimes',
#                       0, ['selected_data'],
#                       [output_merged[1:]],
#                       ['base', 'gate', 'base gate', 'base runtimes', 'gate runtimes', 'base gate runtimes'],
#                       20, 200, output_file='clustering_general/clustering_general_plot_comb_base_gate_runtimes_dec',
#                       show_plot=False,
#                       use_mat_plot=True, use_dash_plot=True)
#
# plot_cpar2_comparison(['clustering_general/clustering_general_par2'], 'vbs_sbs/vbs', 'vbs_sbs/sbs',
#                       'CPar2 scores of different cluster algorithms using combinations of base, gate and runtimes',
#                       0, ['cluster_algorithm', 'selected_data'],
#                       [['KMEANS', 'AFFINITY', 'MEANSHIFT', 'SPECTRAL', 'AGGLOMERATIVE', 'OPTICS', 'GAUSSIAN', 'DBSCAN',
#                         'BIRCH'], output_merged[1:]],
#                       ['K-Means', 'Affintiy Propagation', 'Meanshift', 'Spectral Clustering', 'Agglomerative', 'OPTICS',
#                        'Gaussian', 'DBSCAN', 'BIRCH'],
#                       20, 200, output_file='clustering_general/clustering_general_plot_algo_base_gate_runtimes_dec',
#                       show_plot=False,
#                       use_mat_plot=True, use_dash_plot=True)


# plot_cpar2_comparison(['scaling_standardscaler/standardscaler_linearscaler_clustering_par2'], 'vbs_sbs/vbs', 'vbs_sbs/sbs',
#                      'Comparison of CPar2 scores between Linear Scaling and Standard Scaling for clusterings with less than 20 clusters',
#                      'scaling_algorithm',
#                       ['SCALEMINUSPLUS1', 'STANDARDSCALER'], ['Linear Scaling to [-1,+1]', 'Standard Scaling'], 20, 100,
#                       output_file='scaling_standardscaler/standardscaler_linearscaler_clustering_par2', show_plot=True)

# single feature clustering
# ----------------------------------------------------------------------------------------------------------------------


def plot_best_cluster_comparison(input_files_par2_scores, plot_description,
                                 highlight_index,
                                 param_names,
                                 value_lists, label_list, max_cluster_amount, min_cluster_amount, cutoff,
                                 output_file='', show_plot=False,
                                 use_mat_plot=True, use_dash_plot=False):
    x_label = 'Best clusters sorted by Par2'
    y_label = 'Par2 Score (s)'

    data = []
    for input_file in input_files_par2_scores:
        data = data + read_json(input_file)

    cluster_list = []
    for evaluation in data:
        cluster_size = Counter(evaluation['clustering'])
        for cluster, size in cluster_size.items():
            if size >= min_cluster_amount:
                cluster_list.append(dict(evaluation, **{'cluster_idx': cluster,
                                                        'cluster_cpar2': evaluation['par2'][1][str(cluster)][0][0][1],
                                                        'cluster_size': size,
                                                        'cluster_solver': evaluation['par2'][1][str(cluster)][0][0][
                                                            0]}))

    sorted_cluster_list = sorted(cluster_list, key=lambda d: d['cluster_cpar2'])
    export_list = []

    value_dict = {}
    for elem in value_lists[highlight_index]:
        value_dict[str(elem)] = []

    keys = []
    text = []
    counter = 0
    for cluster_eval in sorted_cluster_list:
        if len(keys) >= cutoff:
            break

        eval_is_in_graph = True
        for idx, value in enumerate(param_names):
            if cluster_eval['settings'][value] not in value_lists[idx]:
                eval_is_in_graph = False
                break

        if eval_is_in_graph and len(cluster_eval['clusters']) <= max_cluster_amount:

            for key, value in value_dict.items():
                if str(key) == str(cluster_eval['settings'][param_names[highlight_index]]):
                    value_dict[str(key)].append(cluster_eval['cluster_cpar2'])
                else:
                    value_dict[str(key)].append(0)
            keys.append(counter)
            counter = counter + 1

            text_string = util.add_line_breaks_to_text(str(cluster_eval['settings']), ',', 5)

            text.append(text_string + ' cluster idx: ' + str(cluster_eval['cluster_idx']) +
                        ' cluster size: ' + str(cluster_eval['cluster_size']) +
                        ' solver: ' + str(cluster_eval['cluster_solver']) +
                        ' id: ' + str(cluster_eval['id']))

            export_list.append((cluster_eval['id'], cluster_eval['cluster_idx']))

    if output_file != '':
        write_json(output_file, export_list)

    if use_mat_plot:

        dpi = 150
        barwidth = 1.2
        plt.figure(figsize=(1200 / dpi, 500 / dpi), dpi=dpi)
        # plt.bar(keys, vbs_best, label=vbs_best_label, width=barwidth)
        # plt.bar(keys, vbs_worst, label=vbs_worst_label, width=barwidth)
        for idx, elem in enumerate(value_lists[highlight_index]):
            plt.bar(keys, value_dict[str(elem)], label=label_list[idx], width=barwidth)

        plt.xlabel(x_label)
        plt.ylabel(y_label)
        ax = plt.subplot(111)
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.title(plot_description)

        if output_file != '':
            plt.savefig(os.environ['EXPPATH'] + output_file + '.svg')
        if show_plot:
            plt.show()

    if use_dash_plot:
        chart_data = [
            # go.Bar(name=vbs_best_label, x=keys, y=vbs_best, hovertext=text),
            # go.Bar(name=vbs_worst_label, x=keys, y=vbs_worst, hovertext=text)
        ]

        for idx, key in enumerate(value_lists[highlight_index]):
            chart_data.append(
                go.Bar(name=label_list[idx], x=keys, y=value_dict[str(key)], hovertext=text)
            )

        fig = go.Figure(layout=go.Layout(
            title=go.layout.Title(text=plot_description)
        ),
            data=chart_data)

        fig.update_layout(barmode='stack', bargap=0)
        fig.update_xaxes(title_text=x_label)
        fig.update_yaxes(title_text=y_label)

        if output_file != '':
            exportFigure.export_plot_as_html(fig, output_file)
        if show_plot:
            fig.show()


# temp_solver_features = DatabaseReader.FEATURES_SOLVER.copy()
# temp_solver_features.pop(14)
# temp_solver_features.pop(7)
# input_dbs = [DatabaseReader.FEATURES_BASE, DatabaseReader.FEATURES_GATE, temp_solver_features]
# output = sum([list(map(list, itertools.combinations(input_dbs, i))) for i in range(len(input_dbs) + 1)], [])
# output_merged = []
# for combination in output:
#     comb = []
#     for elem in combination:
#         comb = comb + elem
#     output_merged.append(comb)
#
# plot_best_cluster_comparison(['clustering_general/clustering_general_par2'],
#                       'Par2 scores of the best clusters in clusterings using combinations of base, gate, runtimes',
#                       0, ['selected_data'],
#                       [output_merged[1:]],
#                       ['base', 'gate', 'runtimes', 'base gate', 'base runtimes', 'gate runtimes', 'base gate runtimes'],
#                       100, 100, 5000, output_file='clustering_best_cluster/clustering_best_cluster_all_5000',
#                       show_plot=False,
#                       use_mat_plot=True, use_dash_plot=True)

# ----------------------------------------------------------------------------------------------------------------------


# Draws a plot of the mean par2 score and the par2 score of the clusters as blobs (with respective size to the
# amount of instances in the cluster
# input_file: The file to read the par2 scores from
# plot_description: The title/description of the plot
# settings_dict: The settings of the clustering we want to plot (for how the dict should look, see the comment on
# the collect_evaluation function
# iter_param: The parameter to iterate on the x axis (must be sortable)
# iter_param_label: The label of the x axis
# output_file: The filename of the exported html (no export if equal to '')
# show_plot: If the plot should be opened in the browser after running the function
# show_best_solver: If the best solver should be written above each cluster blob (without hovering)
def plot_par2(input_file, plot_description, settings_dict, iter_param, iter_param_label, output_file='',
              show_plot=False,
              show_best_solver=False):
    data, diff = collect_evaluation(input_file, settings_dict)
    if iter_param not in diff:
        raise AssertionError('Iteration Parameter not in selected evaluations')

    sorted_data = sorted(data, key=lambda d: d['settings'][iter_param])

    p_keys_par2 = []
    p_value_par2 = []
    p_keys_amount = []
    p_value_amount = []
    p_size_amount = []
    p_text_amount = []
    for evaluation in sorted_data:
        iter_value = evaluation['settings'][iter_param]
        p_keys_par2.append(iter_value)
        p_value_par2.append(evaluation['par2'][0])
        p_keys_amount = p_keys_amount + [iter_value] * len(evaluation['clusters'])
        p_value_amount = p_value_amount + [evaluation['par2'][1][str(cluster)][-1] for cluster in
                                           evaluation['clusters']]
        cluster_size = Counter(evaluation['clustering'])
        p_size_amount = p_size_amount + [cluster_size[cluster] for cluster in evaluation['clusters']]
        p_text_amount = p_text_amount + [str(evaluation['par2'][1][str(cluster)][0][0:5]) + " cluster_size: " +
                                         str(cluster_size[cluster]) for cluster in
                                         evaluation['clusters']]

    fig = go.Figure(layout=go.Layout(
        title=go.layout.Title(text=plot_description)
    ))

    mode = 'markers'
    if show_best_solver:
        mode = 'markers+text'

    fig.add_trace(go.Scatter(x=list(p_keys_par2), y=p_value_par2, mode='lines', name='Mean par2 score'))
    fig.add_trace(go.Scatter(x=p_keys_amount, y=p_value_amount, mode=mode,
                             marker=dict(size=[np.sqrt(item / np.pi) * 2 for item in p_size_amount]),
                             text=[elem[0][0] for elem in p_text_amount], name='Par2 score per cluster',
                             hovertext=p_text_amount, textposition='top center'))

    fig.update_xaxes(title_text=iter_param_label)
    fig.update_yaxes(title_text='par2 score')

    if output_file != '':
        exportFigure.export_plot_as_html(fig, output_file)

    if show_plot:
        fig.show()


# Examples

# plot_par2('par2', 'Par2 scores for each k in [1,9] with kmeans using base.db.',
#           {"scaling_algorithm": ["SCALEMINUSPLUS1"], "scaling_technique": ["NORMALSCALE"],
#            "selection_algorithm": ["NONE"],
#            "selected_data": [["base"]], "cluster_algorithm": ["KMEANS"], "seed": [0]}, "n_clusters_k_means", 'k',
#           show_plot=True)
#
# plot_par2('par2', 'Par2 scores for each eps with dbscan using base.db.',
#           {"scaling_algorithm": ["SCALEMINUSPLUS1"], "scaling_technique": ["NORMALSCALE"],
#            "selection_algorithm": ["NONE"],
#            "selected_data": [["base"]], "cluster_algorithm": ["DBSCAN"], 'min_samples_dbscan': [3]},
#            "eps_dbscan", 'eps',
#           show_plot=True)

# ----------------------------------------------------------------------------------------------------------------------


# plots the mutual information of the different db combinations
# input_file: the file containing the mutual information data
# plot_description: The title/description of the plot
# settings_dict: The experiments settings that should be used
# iter_param: The parameter name that should be iterated on the x axis
# iter_param_label: The label of the x axis
# mutual_info_param: The parameter name of the mutual information in the input_file
# mutual_info_label: The label of the y axis (mutual info)
# output_file: The filename of the exported html (no export if equal to '')
# show_plot: If the plot should be opened in the browser after running the function
def plot_mutual_info(input_file, plot_description, settings_dict, iter_param, iter_param_label, mutual_info_param,
                     mutual_info_label, output_file='',
                     show_plot=False):
    data, diff = collect_evaluation(input_file, settings_dict)
    if iter_param not in diff:
        raise AssertionError('Iteration Parameter not in selected evaluations')

    combinations = settings_dict['selected_data']

    fig = go.Figure(layout=go.Layout(
        title=go.layout.Title(text=plot_description)
    ))
    fig.update_xaxes(title_text=iter_param_label)
    fig.update_yaxes(title_text=mutual_info_label)

    for comb in combinations:
        comb_data = []
        for evaluation in data:
            if evaluation['settings']['selected_data'] == comb:
                comb_data.append(evaluation)

        comb_data = sorted(comb_data, key=lambda d: d['settings'][iter_param])
        keys = []
        value = []
        for evaluation in comb_data:
            keys.append(evaluation['settings'][iter_param])
            value.append(evaluation[mutual_info_param])

        fig.add_trace(go.Scatter(x=keys, y=value, mode='lines', name=" ".join(str(x) for x in comb)))

    if output_file != '':
        exportFigure.export_plot_as_html(fig, output_file)

    if show_plot:
        fig.show()


# Example:

# plot_mutual_info('normalized_mututal_information_family',
#                  'Normalized mutual information between clustering and the cluster induced by the families',
#                  {"scaling_algorithm": ["SCALEMINUSPLUS1"], "scaling_technique": ["NORMALSCALE"],
#                   "selection_algorithm": ["NONE"],
#                   "selected_data": [["base"], ['gate'], ['solver'], ['base', 'gate'], ['base', 'solver'],
#                                     ['gate', 'solver'], ['base', 'gate', 'solver']],
#                   "cluster_algorithm": ["KMEANS"], "seed": [0]}, "n_clusters_k_means", 'k',
#                  'normalized_mutual_information_family', 'Normalized Mutual Information',
#                  show_plot=True
#                  )

# ----------------------------------------------------------------------------------------------------------------------


# Creates a scatter plot of the selected plot, where the size of the points is the cluster size and the color
# shows to which cluster it belongs.
# Additional information are the value of the two selected parameters for x and y as well as the best solver and it's
# time to solver the particular instance
# input_file: The file which contains the clustering
# db_instance: A DbInstance,
# plot_description: The description/title of the plot
# settings_dict: Used to select the clustering you want to plot. If there are multiple clusterings that fit the
# settings the first one is used
# x_param: The parameter of x-axis. Must be a parameter of base.db, gate.db or runtimes.db
# x_param_label: What name the x_param should be given in the plot
# y_param: The parameter of y-axis. Must be a parameter of base.db, gate.db or runtimes.db
# y_param_label: What name the y_param should be given in the plot
# output_file: The filename of the exported html (no export if equal to '')
# show_plot: If the plot should be opened in the browser after running the function
def plot_clustering(input_file, db_instance: DbInstance, plot_description, settings_dict, x_param, x_param_label,
                    y_param,
                    y_param_label, output_file='',
                    show_plot=False):
    collected, diff = collect_evaluation(input_file, settings_dict)
    selected_cluster = collected[0]

    dataset, dataset_wh, dataset_f = db_instance.get_complete_dataset()
    x_param_idx = dataset_f.index(x_param)
    y_param_idx = dataset_f.index(y_param)

    best_solver_time = [min(elem) for elem in db_instance.solver_wh]
    best_solver = [db_instance.solver_f[argmin(elem)] for elem in db_instance.solver_wh]

    scatter_values = util.rotateNestedLists(dataset_wh)

    dataframe_dict = {
        x_param_label: scatter_values[x_param_idx],
        y_param_label: scatter_values[y_param_idx],
        'cluster': selected_cluster['clustering'],
        'best_solver_time': best_solver_time,
        'best_solver': best_solver
    }

    df = pd.DataFrame(dataframe_dict)
    df["cluster"] = df["cluster"].astype(str)
    fig = px.scatter(df, x=x_param_label, y=y_param_label, color='cluster', size='best_solver_time',
                     hover_data=['best_solver'], title=plot_description)

    fig.update_xaxes(title_text=x_param_label)
    fig.update_yaxes(title_text=y_param_label)

    if output_file != '':
        exportFigure.export_plot_as_html(fig, output_file)

    if show_plot:
        fig.show()


# Example:

# plot_clustering('basic_search_all_cluster_algorithms', DbInstance(),
#                 'Clustering',
#                 {"scaling_algorithm": ["SCALEMINUSPLUS1"], "scaling_technique": ["NORMALSCALE"],
#                  "selection_algorithm": ["NONE"], "selected_data": [["base"]], "scaling_k_best": [3],
#                  "cluster_algorithm": ["KMEANS"], "seed": [0], "n_clusters_k_means": [5]},
#                 'n_vars', 'n_vars', 'n_gates', 'n_gates', '', show_plot=True)

# ----------------------------------------------------------------------------------------------------------------------

# generates a pie chart for each given cluster where the distribution of the families in the cluster is shown
# input_file: The file containing the clusterings
# db_instance
# cluster_file: File containing the clustering ids and clustering idx for which the pie charts should be drawn
def plot_single_cluster_distribution_family(input_file, db_instance: DbInstance, cluster_file,
                                            output_file='',
                                            show_plot=False, use_mat_plot=True):
    data = read_json(input_file)
    cluster_data = read_json(cluster_file)
    clustering_id_list = [item[0] for item in cluster_data]
    cluster_idx_list = [item[1] for item in cluster_data]

    columns = 10
    row = ceil(len(cluster_idx_list) / columns)
    keys_all = list(set([item[0] for item in db_instance.family_wh]))
    keys_color_dict = {}
    for idx, key in enumerate(keys_all):
        keys_color_dict[key] = util.random_color()

    keys_list = []
    colors_list = []
    values_list = []
    for clustering_id, cluster_idx in zip(clustering_id_list, cluster_idx_list):

        instance = None
        for elem in data:
            if elem['id'] == clustering_id:
                instance = elem
                break

        family_dict = {}
        size = 0
        for cluster_elem in instance['clustering']:
            if cluster_elem == cluster_idx:
                size = size + 1
                current_family = db_instance.family_wh[cluster_elem][0]
                if current_family in family_dict:
                    family_dict[current_family] = family_dict[current_family] + 1
                else:
                    family_dict[current_family] = 1

        values = []
        keys = []
        colors = []
        for family in family_dict:
            values.append(family_dict[family] / size)
            keys.append(family)
            colors.append(keys_color_dict[family])

        values_list.append(values)
        keys_list.append(keys)
        colors_list.append(colors)

    if use_mat_plot:
        dpi = 40

        plt.style.use('ggplot')
        fig, axes = plt.subplots(nrows=row, ncols=columns, figsize=(1200 / dpi, 500 / dpi), dpi=dpi)

        for ax, values, keys, colors in zip(axes.flat, values_list, keys_list, colors_list):
            ax.pie(values, labels=keys, autopct='%.2f', colors=colors)
            ax.set(ylabel='', aspect='equal')

        if output_file != '':
            plt.savefig(os.environ['EXPPATH'] + output_file + '.svg')

        if show_plot:
            plt.show()


    # if use_dash_plot:
    #
    #     fig = go.Figure(data=[go.Pie(labels=keys,
    #                                  values=values)],
    #                     layout=go.Layout(
    #                         title=go.layout.Title(
    #                             text=plot_description))
    #                     )
    #
    #     if output_file != '':
    #         exportFigure.export_plot_as_html(fig, output_file)
    #
    #     if show_plot:
    #         fig.show()
    #

# ----------------------------------------------------------------------------------------------------------------------

# plots for a single cluster how the solvers are distributed in ranks, by counting the occurrence of each solver in
# each rank (determined by order of solvers sorted by runtime for the instance in the cluster) and scaling it to [0,1]
# input_file: File containing the clustering
# clustering_id: The id of the clustering to read
# cluster_idx: The index of the cluster in the clustering
# db_instance: Important: Init with the correct solver features, that were used in the clustering
# output_file: Name of the generated graph file
# show_plot: Should the plot be shown after executing?
def plot_cluster_best_solver_distribution(input_file, clustering_id, cluster_idx, db_instance:DbInstance,
                                          output_file='', show_plot=True):

    # don't take the solvers from DatabaseReader.FEATURES_SOLVERS if not all solvers where used in the clustering
    # because glucose_syrup and yalsat were removed, we take it directly from the reduced db_instance
    key_names = db_instance.solver_f

    data = read_json(input_file)
    clustering = None
    for elem in data:
        if elem['id'] == clustering_id:
            clustering = elem
            break

    instances_in_cluster = []
    size = 0
    for i, instance in enumerate(clustering['clustering']):
        if instance == cluster_idx:
            size = size + 1
            zipped_solvers = list(zip(db_instance.solver_wh[i], key_names))
            instances_in_cluster.append(sorted(zipped_solvers, key=lambda x: x[0]))

    # contains arrays with the amounts of the solvers for each rank
    ranks = []

    solver_amount = len(db_instance.solver_f)

    for i in range(solver_amount):
        rank_value = []
        rank_dict = {}
        for solver in key_names:
            rank_dict[solver] = 0
        for instance in instances_in_cluster:
            rank_dict[instance[i][1]] = rank_dict[instance[i][1]] + 1
        for solver in key_names:
            value = float(rank_dict[solver]) / float(size)
            rank_value.append(value)
        ranks.append(rank_value)

    dpi = 90
    plt.figure(figsize=(1200 / dpi, 600 / dpi), dpi=dpi)

    for i in range(solver_amount):
        bottom = [0] * len(key_names)
        for j in range(i):
            bottom = np.array(bottom) + np.array(ranks[j])
        plt.bar(key_names, ranks[i], label=0, bottom=bottom)

    plt.title(clustering['par2'][1][str(cluster_idx)][0][0][0])
    y_pos = range(len(key_names))
    plt.xticks(y_pos, key_names, rotation=90)
    plt.subplots_adjust(bottom=0.3)

    if output_file != '':
        plt.savefig(os.environ['EXPPATH'] + output_file + '.svg')

    if show_plot:
        plt.show()



