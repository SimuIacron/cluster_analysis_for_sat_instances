from sklearn.metrics import normalized_mutual_info_score

from DataAnalysis import scoring_util
from DataAnalysis.scoring_modular import score, f1_par2, f2_par2_cluster, f3_weigh_with_cluster_size
from DataFormats.DbInstance import DbInstance
from Experiment_pipeline.run_experiments import read_json_file, write_cluster_results_to_json
import multiprocessing as mp


def run_evaluation_normalized_mutual_info(filename):

    db_instance = DbInstance()
    cluster_results = read_json_file(filename)

    family_int = scoring_util.convert_families_to_int(db_instance.family_wh)

    pool = mp.Pool(mp.cpu_count())
    result_family = [{'id': entry['id'], 'cluster_algorithm': entry['settings']['cluster_algorithm'],
                      'clusters': entry['clusters'],
                      'norm_mut_info': pool.apply(normalized_mutual_info_score, args=(family_int, entry['clustering']))}
                     for entry in cluster_results]

    k = 15
    result_family_k = []
    for entry in result_family:
        if len(entry['clusters']) < k:
            result_family_k.append(entry)

    sorted_results_family = sorted(result_family_k, key=lambda d: d['norm_mut_info'], reverse=True)


def run_evaluation_score(filename):
    db_instance = DbInstance()
    cluster_results = read_json_file(filename)
    pool = mp.Pool(mp.cpu_count())
    result = [{'id': entry['id'], 'cluster_algorithm': entry['settings']['cluster_algorithm'],
                      'clusters': entry['clusters'],
                      'score': pool.apply(score, args=(entry['clusters'], entry['clustering'], db_instance, 5000,
                                                       f1_par2, f2_par2_cluster, f3_weigh_with_cluster_size))}
                     for entry in cluster_results]

    write_cluster_results_to_json('par2_score', result)


def par2_sort(filename):
    cluster_results = read_json_file(filename)[0]
    sorted_results_family = sorted(cluster_results, key=lambda d: d['score'][0], reverse=True)
    pass

if __name__ == '__main__':
    # run_evaluation_score('basic_search_all_cluster_algorithms_id')
    par2_sort('par2_score')

