import os
from functools import partial
from time import time

from sklearn.metrics import normalized_mutual_info_score

from run_plotting_clusters import export_clusters_sorted_best
from util_scripts import DatabaseReader
import run_experiments
from DataAnalysis.Evaluation import scoring_util
from DataAnalysis.Evaluation.scoring_modular import score, f1_par2, f2_par2_cluster, f3_weigh_with_cluster_size, \
    score_virtual_best_solver, score_single_best_solver, f3_weigh_with_cluster_size_n_best_cluster
from DataFormats.DbInstance import DbInstance
from run_experiments import read_json, write_json, read_json_temp, append_json_temp
import multiprocessing as mp


# WARNING: Needs to execute in __main__ because it contains multithreading
# Generic method to run parallel evaluations
# input_file: Filename of file to read clustering from
# output_file: Filename of file to write the result to
# func_eval: The function used for scoring the clusters. First argument needs to take in one element of the
# datastructures of the
# input file, after that other arguments can be passed with args
# dict_name: Name of the key, the result value should have in the output file
# args: Extra arguments given to the func_eval. Note: Func_eval can only have 2 params, entry and args,
# however args can be any datatype
# num_cores: Number of cpu cores that should be used in parallel
# (uses max available cores, if cores is higher than available cores)
def run_evaluation(input_file, output_file, func_eval, dict_name, args, num_cores):
    t_start = time()
    cluster_results = read_json(input_file)

    temp_filename = output_file + '_temp'

    continue_evaluation = os.path.exists(run_experiments.cluster_result_path + temp_filename + '.txt')
    print('Continuing: ' + str(continue_evaluation))

    if num_cores > mp.cpu_count():
        num_cores = mp.cpu_count()

    print('Available cores: ' + str(mp.cpu_count()))
    print('Cores used: ' + str(num_cores))

    id_list = []
    if continue_evaluation:
        current_result_list = read_json_temp(temp_filename)
        id_list = [item['id'] for item in current_result_list]

    pool = mp.Pool(num_cores)

    result_objects = []
    # asynchronous evaluation of experiments (order is restored with sorting afterwards)
    for idx, entry in enumerate(cluster_results):
        if continue_evaluation and entry['id'] in id_list:
            continue

        callback_function = partial(func_callback, filename=temp_filename, entry=entry, dict_name=dict_name)
        result = pool.apply_async(func_eval, args=(entry, args), callback=callback_function)
        result_objects.append(result)

    [result.wait() for result in result_objects]
    finished = read_json_temp(temp_filename)

    # evaluation_result = [dict(entry, **{dict_name: result.get()}) for (entry, result) in result_objects]

    # sort evaluation after id before writing it to file
    write_json(output_file, sorted(finished, key=lambda d: d['id']))

    t_stop = time()
    print('Evaluation took %f' % (t_stop - t_start))


def func_callback(result, filename, entry, dict_name):
    append_json_temp(filename, dict(entry, **{dict_name: result}))


# -- Normalized mutual information ------------------------------------------------------------------------------------

# normalized mutual information evaluation function for the parallel execution
def eval_normalized_mutual_info(entry, args):
    return normalized_mutual_info_score(args, entry['clustering'])


# Calculates the normalized mutual information of the clustering and the cluster induced by the families of all given
# experiments
# input_file: The file with the experiments to calculate the par2 score of
# output_file: The file to write the results to
# cores: Number of cpu cores that should be used in parallel
# (uses max available cores, if cores is higher than available cores)
def run_evaluation_normalized_mutual_info_family(input_file, output_file, db_instance: DbInstance, num_cores):
    family_int = scoring_util.convert_families_to_int(db_instance.family_wh)

    run_evaluation(input_file, output_file, eval_normalized_mutual_info, 'normalized_mutual_information_family',
                   family_int, num_cores)


# Calculates the normalized mutual information of the clustering and the cluster induced by the best solvers of all
# given experiments
# input_file: The file with the experiments to calculate the par2 score of
# output_file: The file to write the results to
# num_cores: Number of cpu cores that should be used in parallel
# (uses max available cores, if cores is higher than available cores)
def run_evaluation_normalized_mutual_info_best_solver(input_file, output_file, db_instance: DbInstance, num_cores):
    best_solver_int = scoring_util.convert_best_solver_int(db_instance)

    run_evaluation(input_file, output_file, eval_normalized_mutual_info, 'normalized_mutual_information_best_solver',
                   best_solver_int, num_cores)


# Calculates the normalized mutual information of the clustering and the cluster induced by sat/unsat of all given
# experiments
# input_file: The file with the experiments to calculate the par2 score of
# output_file: The file to write the results to
# num_cores: Number of cpu cores that should be used in parallel
# (uses max available cores, if cores is higher than available cores)
def run_evaluation_normalized_mutual_info_un_sat(input_file, output_file, db_instance: DbInstance, num_cores):
    un_sat_int = scoring_util.convert_un_sat_to_int(db_instance.un_sat_wh)

    run_evaluation(input_file, output_file, eval_normalized_mutual_info, 'normalized_mutual_information_un_sat',
                   un_sat_int, num_cores)


# -- Par2 score --------------------------------------------------------------------------------------------------------

# par2 evaluation function for the parallel execution
def eval_par2_score(entry, args):
    return score(entry['clusters'], entry['clustering'], args[0], args[1], f1_par2, f2_par2_cluster,
                 f3_weigh_with_cluster_size)


# Calculates the par2 score of all given experiments
# input_file: The file with the experiments to calculate the par2 score of
# output_file: The file to write the results to
# num_cores: Number of cpu cores that should be used in parallel
# (uses max available cores, if cores is higher than available cores)
def run_evaluation_par2_score(input_file, output_file, db_instance: DbInstance, num_cores):
    run_evaluation(input_file, output_file, eval_par2_score, 'par2', (db_instance, DatabaseReader.TIMEOUT), num_cores)


# Calculates the Par2 score if the best solver would be used for each instance
# output_file: The file to write the results to
def run_evaluation_par2_vbs(output_file, db_instance: DbInstance):
    final_score, cluster_score_dict = score_virtual_best_solver(db_instance, DatabaseReader.TIMEOUT, f1_par2,
                                                                f2_par2_cluster, f3_weigh_with_cluster_size)
    write_json(output_file, [final_score, cluster_score_dict])


# Calculate the Par2 score if the best solver over all instances would be used
# output_file: The file to write the results to
def run_evaluation_par2_sbs(output_file, db_instance: DbInstance):
    final_score, cluster_score_dict = score_single_best_solver(db_instance, DatabaseReader.TIMEOUT, f1_par2,
                                                               f2_par2_cluster, f3_weigh_with_cluster_size)
    write_json(output_file, [final_score, cluster_score_dict])


def run_evaluation_par2_vbs_n_best(output_file, db_instance: DbInstance, n):
    func_weight = \
        lambda yhat, clusters, db1, timeout, dict1: f3_weigh_with_cluster_size_n_best_cluster(yhat, clusters, db1,
                                                                                              timeout, dict1, n, False)
    final_score, cluster_score_dict = score_virtual_best_solver(db_instance, DatabaseReader.TIMEOUT, f1_par2,
                                                                f2_par2_cluster, func_weight)
    write_json(output_file, [final_score, cluster_score_dict])


def run_evaluation_par2_vbs_n_worst(output_file, db_instance: DbInstance, n):
    func_weight = \
        lambda yhat, clusters, db1, timeout, dict1: f3_weigh_with_cluster_size_n_best_cluster(yhat, clusters, db1,
                                                                                              timeout, dict1, n, True)
    final_score, cluster_score_dict = score_virtual_best_solver(db_instance, DatabaseReader.TIMEOUT, f1_par2,
                                                                f2_par2_cluster, func_weight)
    write_json(output_file, [final_score, cluster_score_dict])


def run_evaluation_par2_sbs_n_best(output_file, db_instance: DbInstance, n):
    func_weight = \
        lambda yhat, clusters, db1, timeout, dict1: f3_weigh_with_cluster_size_n_best_cluster(yhat, clusters, db1,
                                                                                              timeout, dict1, n, False)
    final_score, cluster_score_dict = score_single_best_solver(db_instance, DatabaseReader.TIMEOUT, f1_par2,
                                                               f2_par2_cluster, func_weight)
    write_json(output_file, [final_score, cluster_score_dict])


def run_evaluation_par2_sbs_n_worst(output_file, db_instance: DbInstance, n):
    func_weight = \
        lambda yhat, clusters, db1, timeout, dict1: f3_weigh_with_cluster_size_n_best_cluster(yhat, clusters, db1,
                                                                                              timeout, dict1, n, True)
    final_score, cluster_score_dict = score_single_best_solver(db_instance, DatabaseReader.TIMEOUT, f1_par2,
                                                               f2_par2_cluster, func_weight)
    write_json(output_file, [final_score, cluster_score_dict])


# ----------------------------------------------------------------------------------------------------------------------


if __name__ == '__main__':
    cores = 20

    temp_solver_features = DatabaseReader.FEATURES_SOLVER.copy()
    temp_solver_features.pop(14)
    temp_solver_features.pop(7)
    input_dbs = [DatabaseReader.FEATURES_BASE, DatabaseReader.FEATURES_GATE, temp_solver_features]
    features = []
    for feature_vector in input_dbs:
        features = features + feature_vector

    db = DbInstance(features)

    run_evaluation_par2_score('general_clustering_6_linearscaler', 'general_clustering_6_linearscaler_par2', db, cores)

    export_clusters_sorted_best('general_clustering_6_linearscaler_par2', 'general_clustering_6_linearscaler_clusters')

    # run_evaluation_par2_sbs_n_best('sbs_100_best', db, 100)
    # run_evaluation_par2_sbs_n_worst('sbs_100_worst', db, 100)

    # run_evaluation_par2_vbs('vbs_without_glucose_syrup_yalsat', db)
    # run_evaluation_par2_bss('bss', db)
    # run_evaluation_normalized_mutual_info_family('basic_search_all_cluster_algorithms', 'mutual_info_family', db, cores)
    # run_evaluation_normalized_mutual_info_best_solver('basic_search_all_cluster_algorithms', 'mutual_info_best_solver',
    #                                                   db, cores)
    # run_evaluation_normalized_mutual_info_un_sat('basic_search_all_cluster_algorithms', 'mutual_info_un_sat', db, cores)
