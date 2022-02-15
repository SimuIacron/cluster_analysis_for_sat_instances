from collections import Counter
from DataAnalysis.Evaluation import scoring
from DataFormats.DbInstance import DbInstance


# General function to score clusterings, where the parameters are useful
# - clusters: List of the cluster indexes e.g. [0,1,2]
# - yhat: Mapping of instance to clustering e.g. [2,1,0,0,2,1,1]
# - db_instance: Contains the data of the dbs
# - timeout value
# - func_select_best_solver(yhat, cluster, db_instance, timeout): best_solver:
# Calculates for one cluster which solver is the best (with it's metric) and returns the name of the best solver
# - func_score_single_cluster(yhat, cluster, db_instance, timeout, best_solver): cluster_score:
# Calculates the score for a single cluster with the best solver selected before and returns it's score
# - func_score_complete_clustering(yhat, clusters, db_instance, timeout, cluster_score_dict): final_score:
# Takes all cluster score and calculates the final score form it
def score(clusters, yhat, db_instance: DbInstance, timeout,
          func_sort_solvers_after_best, func_score_single_cluster, func_score_complete_clustering):
    cluster_score_dict = {}
    for cluster in clusters:
        sorted_solvers = func_sort_solvers_after_best(yhat, cluster, db_instance, timeout)
        cluster_score = func_score_single_cluster(yhat, cluster, db_instance, timeout, sorted_solvers[0][0])
        cluster_score_dict[cluster] = (sorted_solvers, cluster_score)

    final_score = func_score_complete_clustering(yhat, clusters, db_instance, timeout, cluster_score_dict)

    return final_score, cluster_score_dict


# wrapper method to score virtual best solver
def score_virtual_best_solver(db_instance: DbInstance, timeout,
                              func_sort_solvers_after_best, func_score_single_cluster, func_score_complete_clustering):
    instance_amount = len(db_instance.solver_wh)
    return score(range(instance_amount), range(instance_amount), db_instance, timeout,
                 func_sort_solvers_after_best, func_score_single_cluster, func_score_complete_clustering)


# wrapper method to score single best solver
def score_single_best_solver(db_instance: DbInstance, timeout,
                             func_sort_solvers_after_best, func_score_single_cluster, func_score_complete_clustering):
    instance_amount = len(db_instance.solver_wh)
    return score([0], [0] * instance_amount, db_instance, timeout,
                 func_sort_solvers_after_best, func_score_single_cluster, func_score_complete_clustering)


# -- func_sort_solvers_after_best(yhat, cluster, db_instance, timeout) -------------------------------------------------


def f1_relative(yhat, cluster, db_instance: DbInstance, timeout):
    solver_dict = scoring.score_solvers_on_relative_runtime_cluster(yhat, cluster, db_instance, timeout)
    solver_list = []
    for solver, solver_score in solver_dict.items():
        solver_list.append((solver, solver_score))

    solver_list.sort(key=lambda x: x[1])

    return solver_list


def f1_par2(yhat, cluster, db_instance: DbInstance, timeout):
    solver_list = []
    for solver in db_instance.solver_f:
        current_score = f2_par2_cluster(yhat, cluster, db_instance, timeout, solver)
        solver_list.append((solver, current_score))

    solver_list.sort(key=lambda x: x[1])

    return solver_list


# -- func_score_single_cluster(yhat, cluster, db_instance, timeout, best_solver) ---------------------------------------


# calculates the par2 score for a given cluster and solver
def f2_par2_cluster(yhat, cluster, db_instance: DbInstance, timeout, best_solver):
    solver_idx = db_instance.solver_f.index(best_solver)
    cluster_size = 0
    score = 0
    for idx, inst in enumerate(db_instance.solver_wh):
        if cluster == yhat[idx]:
            cluster_size = cluster_size + 1
            running_time = inst[solver_idx]
            if running_time >= timeout:
                running_time = timeout * 2
            score = score + running_time

    if cluster_size == 0:
        return 0
    return score / cluster_size


# -- func_score_complete_clustering(yhat, clusters, db_instance, timeout, cluster_score_dict) --------------------------


def f3_weigh_with_cluster_size(yhat, clusters, db_instance: DbInstance, timeout, cluster_score_dict):
    cluster_sizes = Counter(yhat)
    current_score = 0
    for cluster, (sorted_solvers, cluster_score) in cluster_score_dict.items():
        current_score = current_score + cluster_score * cluster_sizes[cluster]

    final_score = current_score / len(yhat)
    return final_score


def f3_weigh_with_cluster_size_n_best_cluster(yhat, clusters, db_instance: DbInstance, timeout, cluster_score_dict, n, reverse):
    score_list = []

    for cluster, (sorted_solvers, cluster_score) in cluster_score_dict.items():
        score_list.append(cluster_score)

    score_list_sorted = sorted(score_list, reverse=reverse)
    final_score = sum(score_list_sorted[:n]) / len(yhat)
    return final_score
