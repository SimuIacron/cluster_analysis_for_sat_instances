import itertools
import json
import os
import numpy as np

from DataAnalysis import feature_selection, scaling, clustering
from DataFormats.DbInstance import DbInstance

cluster_result_path = os.environ['EXPPATH']


# writes one result as a line to the specified file
def append_to_json(filename, result):
    with open(cluster_result_path + filename + '.txt', 'a') as file:
        json_result = json.dumps(result)
        file.write(json_result + '\n')


# reads all results from the given file and returns a list with dictionaries for each result
def read_json(filename):
    results = []
    with open(cluster_result_path + filename + '.txt', 'r') as file:
        lines = file.readlines()
        for line in lines:
            results.append(json.loads(line))

    return results


# runs the feature selection, selection and clustering algorithm with the given experiments
# and writes the result as a new line (json) into the file given with 'filename'
# the experiment list contains experiments, an experiment is a list of tuples structured as follows:
# [(parameter name, list of possible parameter values)]
# e.g. [('cluster_algorithm', ['KMEANS']), ('seed', [0]), ('n_clusters_k_means', range(1, 10))]
def run_experiments(experiment_list, filename):
    db_instance = DbInstance()

    for experiment_param_list in experiment_list:
        params = []
        param_ranges = []
        for idx, (param, param_range) in enumerate(experiment_param_list):
            params.append(param)
            param_ranges.append(param_range)

        # generate every combination of given parameter values of the experiment
        combinations = list(itertools.product(*param_ranges))
        id_counter = 0
        for comb in combinations:
            print(comb)
            # create a dictionary of the current combination of parameters that is passed to the algorithms
            comb_dict = {}
            for idx, param in enumerate(params):
                comb_dict[param] = comb[idx]

            # execute the algorithms
            db_instance.generate_dataset(comb_dict['selected_data'])
            feature_selected_data = feature_selection.feature_selection(db_instance.dataset_wh, db_instance.dataset_f,
                                                                        db_instance.solver_wh, comb_dict)
            scaled_data = scaling.scaling(feature_selected_data, db_instance.dataset_f, comb_dict)
            (clusters, yhat) = clustering.cluster(scaled_data, comb_dict)

            # write the result together with the parameters of the combination into the given file as json
            result = {'id': id_counter, 'settings': comb_dict, 'clusters': clusters.tolist(), 'clustering': yhat.tolist()}
            id_counter = id_counter + 1
            append_to_json(filename, result)


# test code
input_dbs = ['base', 'gate', 'solver']
output = sum([list(map(list, itertools.combinations(input_dbs, i))) for i in range(len(input_dbs) + 1)], [])
standard_settings = [('scaling_algorithm', ['SCALEMINUSPLUS1']),
                     ('scaling_technique', ['SCALE01']),
                     ('selection_algorithm', ['NONE']),
                     ('selected_data', output[1:])]

exp_kmeans = standard_settings + \
             [('cluster_algorithm', ['KMEANS']),
              ('seed', [0]),
              ('n_clusters_k_means', range(1, 10))]

exp_affinity = standard_settings + \
               [('cluster_algorithm', ['AFFINITY']),
                ('seed', [0]),
                ('damping_aff', np.arange(0.5, 1, 0.1)),
                ('preference_aff', [None]),
                ('affinity_aff', ['euclidean'])]

exp_meanshift = standard_settings + \
                [('cluster_algorithm', ['MEANSHIFT']),
                 ('bandwidth_mean', list(range(1, 10)) + [None])]  # not clear what values are useful

exp_spectral = standard_settings + \
               [('cluster_algorithm', ['SPECTRAL']),
                ('seed', [0]),
                ('n_clusters_spectral', range(1, 10))]

exp_agg = standard_settings + \
          [('cluster_algorithm', ['AGGLOMERATIVE']),
           ('n_clusters_agg', range(1, 10)),
           ('affinity_agg', ['euclidean']),
           ('linkage_agg', ['ward', 'complete', 'average', 'single']),
           ('distance_threshold', [None])]  # not clear what float values useful

exp_optics = standard_settings + \
             [('cluster_algorithm', ['OPTICS']),
              ('min_samples_opt', range(1, 10)),
              ('min_clusters_opt', list(range(1, 10)) + [None])]

exp_gaussian = standard_settings + \
               [('cluster_algorithm', ['GAUSSIAN']),
                ('seed', [0]),
                ('n_components_gauss', range(1, 10))]

exp_birch = standard_settings + \
            [('cluster_algorithm', ['BIRCH']),
             ('threshold_birch', np.arange(0.1, 1, 0.1)),
             ('branching_factor_birch', range(10, 100, 10)),
             ('n_clusters_birch', range(1, 10))]

exp_dbscan = standard_settings + \
             [('cluster_algorithm', ['DBSCAN']),
              ('eps_dbscan', np.arange(0.1, 1, 0.1)),
              ('min_samples_dbscan', range(1, 10, 1))]

# run_experiments([exp_kmeans, exp_meanshift, exp_spectral, exp_agg, exp_optics, exp_gaussian,
#                  exp_birch, exp_dbscan], 'basic_search_all_cluster_algorithms')
