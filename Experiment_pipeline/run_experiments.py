import itertools
import json
import os
from pathlib import Path
from time import time

import numpy as np
import multiprocessing as mp

from DataAnalysis import feature_selection, scaling, clustering
from DataFormats.DbInstance import DbInstance


# - Reading/Writing Json Files -----------------------------------------------------------------------------------------

# the path where the input/output json files are stored
cluster_result_path = os.environ['EXPPATH']


# Writes the inputted data as a json file and appends a next highest number, if file already exists in directory to
# avoid accidentally overwriting old results
# filename: The name of the file to be written
# result: The data structure that is saved in json format (must only contain json convertible data structures!)
def write_json(filename, result):
    path = cluster_result_path + filename + '.txt'
    counter = 0
    # make sure to not overwrite a file
    while Path(path).is_file():
        path = cluster_result_path + filename + '_' + str(counter) + '.html'
        counter = counter + 1

    with open(path, 'w') as file:
        json.dump(result, file)


# Reads the given file and returns the data structure stored in it
# filename: The file to be read
def read_json(filename):
    with open(cluster_result_path + filename + '.txt', 'r') as file:
        lines = file.readline()
        return json.loads(lines)


# - run experiments ----------------------------------------------------------------------------------------------------

# WARNING: Needs to execute in __main__ because it contains multithreading
# runs the feature selection, selection and clustering algorithm with the given experiments
# and writes the result as a new line (json) into the file given with 'filename'
# the experiment list contains experiments, an experiment is a list of tuples structured as follows:
# [(parameter name, list of possible parameter values)]
# e.g. [('cluster_algorithm', ['KMEANS']), ('seed', [0]), ('n_clusters_k_means', range(1, 10))]
# experiment_list: Contains all experiment setups and their parameters
# filename: The name of the file where the finished clustering are stored
# num_cores: Number of cpu cores that should be used in parallel
# (uses max available cores, if cores is higher than available cores)
def run_experiments(experiment_list, filename, num_cores):
    t_start = time()

    db_instance = DbInstance()

    if num_cores > mp.cpu_count():
        num_cores = mp.cpu_count()

    print('Available cores: ' + str(mp.cpu_count()))
    print('Cores used: ' + str(num_cores))

    pool = mp.Pool(num_cores)
    result_objects = []
    id_counter = 0
    for experiment_param_list in experiment_list:
        params = []
        param_ranges = []
        for idx, (param, param_range) in enumerate(experiment_param_list):
            params.append(param)
            param_ranges.append(param_range)

        # generate every combination of given parameter values of the experiment
        combinations = list(itertools.product(*param_ranges))

        for comb in combinations:
            result = pool.apply_async(run_single_experiment, args=(comb, id_counter, db_instance, params))
            result_objects.append(result)
            id_counter = id_counter + 1

    experiment_result = [result.get() for result in result_objects]
    write_json(filename, sorted(experiment_result, key=lambda d: d['id']))

    t_stop = time()
    print('Experiments took %f' % (t_stop - t_start))


# eval function to run a single experiment for the parallel runner
# comb: The values of the current experiment
# exp_id: The id of the experiment
# db_instance: A DbInstance
# params: The parameter names of the current experiment
def run_single_experiment(comb, exp_id, db_instance, params):
    print(str(id) + ' ' + str(comb))
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
    result = {'id': exp_id, 'settings': comb_dict, 'clusters': clusters.tolist(), 'clustering': yhat.tolist()}
    return result


# ----------------------------------------------------------------------------------------------------------------------

# Example

input_dbs = ['base', 'gate', 'solver']
output = sum([list(map(list, itertools.combinations(input_dbs, i))) for i in range(len(input_dbs) + 1)], [])
standard_settings = [('scaling_algorithm', ['SCALEMINUSPLUS1']),
                     ('scaling_technique', ['NORMALSCALE']),
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

if __name__ == '__main__':
    run_experiments([exp_kmeans, exp_meanshift, exp_spectral, exp_agg, exp_optics, exp_gaussian,
                     exp_birch, exp_dbscan], 'basic_search_all_cluster_algorithms', 20)
