
# Filters the list of clusterings based on a list of params and the allowed values of settings
import os
from collections import Counter

import matplotlib.pyplot as plt
from sklearn.metrics import normalized_mutual_info_score, v_measure_score, homogeneity_score, completeness_score

from DataFormats.DbInstance import DbInstance
from run_experiments import read_json
from util_scripts import DatabaseReader
from util_scripts.util import get_combinations_of_databases


def calculate_homogeneity_of_clustering(clustering, db_instance: DbInstance):
    families = [item[0] for item in db_instance.family_wh]
    count = Counter(families)
    family_vector = {}
    for key, item in count.items():
        family_vector[key] = []

    for family in families:
        for key, item in family_vector.items():
            if key == family:
                family_vector[key].append(1)
            else:
                family_vector[key].append(0)

    family_homogeneity = {}
    family_completeness = {}
    for key, item in family_vector.items():
        family_homogeneity[key] = homogeneity_score(item, clustering)
        family_completeness[key] = completeness_score(item, clustering)

    return family_homogeneity, family_completeness


def filter_clustering_settings(clusterings, param_list, param_values_list):

    print('Initial Clusterings: {a}'.format(a=len(clusterings)))

    clusterings_filtered = []
    for clustering in clusterings:
        clustering_settings = clustering['settings']
        fulfills_all_params = True
        for param, param_values in zip(param_list, param_values_list):
            if clustering_settings[param] not in param_values:
                fulfills_all_params = False
                break

        if fulfills_all_params:
            clusterings_filtered.append(clustering)

    print('Remaining instances after setting filtering: {a}'.format(a=len(clusterings_filtered)))
    return clusterings_filtered


# check if the values of a param is unique for each clustering (test before iterating a param)
def check_if_param_is_unique_in_cluster_list(clusterings, param):
    current_values = []
    for clustering in clusterings:
        clustering_param = clustering['settings'][param]
        if clustering_param in current_values:
            return False
        else:
            current_values.append(clustering_param)

    return True


def plot_homogeneity_change(clusterings, param, param_label, scoring_function, scoring_function_label, output_file=''):

    assert check_if_param_is_unique_in_cluster_list(clusterings, param), 'found multiple equal param values'

    sorted_clustering = sorted(clusterings, key=lambda d: d['settings'][param])

    previous_clustering = sorted_clustering[0]
    mutual_info_with_previous = []
    cbs_score = []
    param_list = []
    for clustering in sorted_clustering[1:]:
        mutual_info_with_previous.append(scoring_function(previous_clustering['clustering'], clustering['clustering']))
        cbs_score.append(clustering['par2'][0])
        param_list.append(clustering['settings'][param])
        previous_clustering = clustering

    fig, ax1 = plt.subplots()
    ax1.plot(param_list, mutual_info_with_previous, color='red')
    ax1.set_xlabel(param_label, fontsize=14)
    ax1.set_ylabel(scoring_function_label, color="red", fontsize=14)
    ax1.set_ylim(0, 1)

    ax2 = ax1.twinx()
    ax2.plot(param_list, cbs_score, color='blue')
    ax2.set_ylabel("CBS (s)", color="blue", fontsize=14)
    ax2.set_ylim(3500, 4000)
    plt.tight_layout()

    if output_file != '':
        plt.savefig(os.environ['TEXPATH'] + output_file + '.svg')




