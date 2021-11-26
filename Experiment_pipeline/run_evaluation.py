from time import time

from sklearn.metrics import normalized_mutual_info_score

from DataAnalysis.Evaluation import scoring_util
from DataAnalysis.Evaluation.scoring_modular import score, f1_par2, f2_par2_cluster, f3_weigh_with_cluster_size, \
    score_single_best_solver
from DataFormats.DbInstance import DbInstance
from Experiment_pipeline.run_experiments import read_json, write_json
import multiprocessing as mp


# Generic method to run parallel evaluations
# input_file: Filename of file to read clustering from
# output_file: Filename of file to write the result to
# func_eval: The function used for scoring the clusters. First argument needs to take in one element of the
# datastructures of the
# input file, after that other arguments can be passed with args
# dict_name: Name of the key, the result value should have in the output file
# args: Extra arguments given to the func_eval. Note: Func_eval can only have 2 params, entry and args,
# however args can be any datatype
def run_evaluation(input_file, output_file, func_eval, dict_name, args, cores):
    t_start = time()
    cluster_results = read_json(input_file)

    if cores > mp.cpu_count():
        cores = mp.cpu_count()

    print('Available cores: ' + str(mp.cpu_count()))
    print('Cores used: ' + str(cores))

    pool = mp.Pool(cores)

    result_objects = []
    # asynchronous evaluation of experiments (order is restored with sorting afterwards)
    for entry in cluster_results:
        result = pool.apply_async(func_eval, args=(entry, args))
        result_objects.append((entry, result))

    evaluation_result = [dict(entry, **{dict_name: result.get()}) for (entry, result) in result_objects]

    # sort evaluation after id before writing it to file
    write_json(output_file, sorted(evaluation_result, key=lambda d: d['id']))

    t_stop = time()
    print('Evaluation took %f' % (t_stop - t_start))


# -- Normalized mutual information ------------------------------------------------------------------------------------

def eval_normalized_mutual_info(entry, args):
    return normalized_mutual_info_score(args, entry['clustering'])


def run_evaluation_normalized_mutual_info_family(input_file, output_file, cores):
    db_instance = DbInstance()
    family_int = scoring_util.convert_families_to_int(db_instance.family_wh)

    run_evaluation(input_file, output_file, eval_normalized_mutual_info, 'normalized_mutual_information_family',
                   family_int, cores)


def run_evaluation_normalized_mutual_info_solver(input_file, output_file, cores):
    db_instance = DbInstance()
    solver_int = scoring_util.convert_best_solver_int(db_instance)

    run_evaluation(input_file, output_file, eval_normalized_mutual_info, 'normalized_mutual_information_solver',
                   solver_int, cores)


def run_evaluation_normalized_mutual_info_un_sat(input_file, output_file, cores):
    db_instance = DbInstance()
    un_sat_int = scoring_util.convert_un_sat_to_int(db_instance.un_sat_wh)

    run_evaluation(input_file, output_file, eval_normalized_mutual_info, 'normalized_mutual_information_un_sat',
                   un_sat_int, cores)


# -- Par2 score --------------------------------------------------------------------------------------------------------

def eval_par2_score(entry, args):
    return score(entry['clusters'], entry['clustering'], args[0], args[1], f1_par2, f2_par2_cluster,
                 f3_weigh_with_cluster_size)


def run_evaluation_par2_score(input_file, output_file, cores):
    db_instance = DbInstance()
    run_evaluation(input_file, output_file, eval_par2_score, 'par2', (db_instance, 5000), cores)


def run_evaluation_par2_bss(output_file):
    db_instance = DbInstance()
    final_score, cluster_score_dict = score_single_best_solver(db_instance, 5000, f1_par2, f2_par2_cluster,
                                                               f3_weigh_with_cluster_size)
    write_json(output_file, [final_score, cluster_score_dict])

# ----------------------------------------------------------------------------------------------------------------------


if __name__ == '__main__':
    run_evaluation_par2_score('basic_search_all_cluster_algorithms', 'par2', 20)
    run_evaluation_par2_bss('bss')
    run_evaluation_normalized_mutual_info_family('basic_search_all_cluster_algorithms', 'mutual_info_family', 20)
    run_evaluation_normalized_mutual_info_solver('basic_search_all_cluster_algorithms', 'mutual_info_solver', 20)
    run_evaluation_normalized_mutual_info_un_sat('basic_search_all_cluster_algorithms', 'mutual_info_un_sat', 20)
    pass
