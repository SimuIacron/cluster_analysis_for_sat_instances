from collections import Counter

import numpy as np
import plotly.graph_objects as go

import exportFigure
from Experiment_pipeline.run_experiments import read_json


# gets the evaluations (or experiments) depending on the given settings_dict from the input_file
# the settings dict, needs to have the same structure as the dict stored in "settings" in the input file,
# however instead of values, for each settings list of settings is used, to enable the usage of "or"
# example: In the file the structure of the problem is given as {settings: {"scaling_algorithm": "SCALEMINUSPLUS1" }}
# then the settings dict needs to hve the structure: {"scaling_algorithm": ["SCALEMINUSPLUS1"] }, we can especially use
# {"scaling_algorithm": ["SCALEMINUSPLUS1", "SCALE01"] } to select evaluations with different scaling algorithm

# returns the list of evaluations that fit the dict and a diff which contains all keys, that where missing in the
# settings_dict, but where present in at least one selected evaluation
def collect_evaluation(input_file, settings_dict):
    data = read_json(input_file)[0]
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


def plot_par2(input_file, plot_description, settings_dict, iter_param, iter_param_label, output_file='', show_plot=False,
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
        p_value_amount = p_value_amount + [evaluation['par2'][1][str(cluster)][-1] for cluster in evaluation['clusters']]
        cluster_size = Counter(evaluation['clustering'])
        p_size_amount = p_size_amount + [cluster_size[cluster] for cluster in evaluation['clusters']]
        p_text_amount = p_text_amount + [evaluation['par2'][1][str(cluster)][0][0:5] for cluster in evaluation['clusters']]

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


plot_par2('par2', 'Par2 scores for each k in [1,9] with kmeans using base.db.',
          {"scaling_algorithm": ["SCALEMINUSPLUS1"], "scaling_technique": ["SCALE01"], "selection_algorithm": ["NONE"],
           "selected_data": [["base"]], "cluster_algorithm": ["KMEANS"], "seed": [0]}, "n_clusters_k_means", 'k',
          show_plot=True)

plot_par2('par2', 'Par2 scores for each eps with dbscan using base.db.',
          {"scaling_algorithm": ["SCALEMINUSPLUS1"], "scaling_technique": ["SCALE01"], "selection_algorithm": ["NONE"],
           "selected_data": [["base"]], "cluster_algorithm": ["DBSCAN"], 'min_samples_dbscan': [3]},
          "eps_dbscan", 'eps',
          show_plot=True)
