from sklearn.metrics import normalized_mutual_info_score

from DataAnalysis.Evaluation import scoring_util
from DataAnalysis.Evaluation.scoring_modular import score, f1_par2, f2_par2_cluster, f3_weigh_with_cluster_size
from DataFormats.DbInstance import DbInstance
from Experiment_pipeline.run_experiments import read_json, append_to_json
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
def run_evaluation(input_file, output_file, func_eval, dict_name, args):
    cluster_results = read_json(input_file)
    pool = mp.Pool(mp.cpu_count())

    evaluation_result = [dict(entry, **{dict_name: pool.apply(func_eval, args=(entry, args))})
                         for entry in cluster_results]
    append_to_json(output_file, evaluation_result)


# -- Normalized mutual information ------------------------------------------------------------------------------------

def eval_normalized_mutual_info(entry, args):
    return normalized_mutual_info_score(args, entry['clustering'])


def run_evaluation_normalized_mutual_info_family(input_file, output_file):
    db_instance = DbInstance()
    family_int = scoring_util.convert_families_to_int(db_instance.family_wh)

    run_evaluation(input_file, output_file, eval_normalized_mutual_info, 'normalized_mutual_information_family',
                   family_int)


def run_evaluation_normalized_mutual_info_solver(input_file, output_file):
    db_instance = DbInstance()
    family_int = scoring_util.convert_best_solver_int(db_instance.family_wh)

    run_evaluation(input_file, output_file, eval_normalized_mutual_info, 'normalized_mutual_information_solver',
                   family_int)


def run_evaluation_normalized_mutual_info_un_sat(input_file, output_file):
    db_instance = DbInstance()
    family_int = scoring_util.convert_sat_unsat_to_int(db_instance.family_wh)

    run_evaluation(input_file, output_file, eval_normalized_mutual_info, 'normalized_mutual_information_un_sat',
                   family_int)


# -- Par2 score --------------------------------------------------------------------------------------------------------

def eval_par2_score(entry, args):
    return score(entry['clusters'], entry['clustering'], args[0], args[1], f1_par2, f2_par2_cluster,
                 f3_weigh_with_cluster_size)


def run_evaluation_par2_score(input_file, output_file):
    db_instance = DbInstance()
    run_evaluation(input_file, output_file, eval_par2_score, 'par2', (db_instance, 5000))


# ----------------------------------------------------------------------------------------------------------------------

if __name__ == '__main__':
    run_evaluation_par2_score('basic_search_all_cluster_algorithms_id', 'test2')

    # run_evaluation_score('basic_search_all_cluster_algorithms_id')
    # result = par2_sort('par2_score')
    # result_bss = score_single_best_solver(DbInstance(), 5000, f1_par2, f2_par2_cluster, f3_weigh_with_cluster_size)

    # best_solver = []
    # for key, item in result_bss[1].items():
    #     best_solver.append(item[0][0][0])

    #  counter = Counter(best_solver)

    # keys = []
    # values = []
    # for key, value in counter.items():
    #     keys.append(key)
    #     values.append(value)

    # fig = go.Figure(data=[go.Pie(labels=keys,
    #                              values=values)],
    #                 layout=go.Layout(
    #                     title=go.layout.Title(
    #                        text='Distributions of solvers if each instance uses the best single solver'))
    #                 )
    # exportFigure.export_plot_as_html(fig, 'bss_dist')
    # fig.show()
