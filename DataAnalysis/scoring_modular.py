from collections import Counter

from DataAnalysis import scoring
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
def score(clusters, yhat, db_instance:DbInstance, timeout,
          func_select_best_solver, func_score_single_cluster, func_score_complete_clustering):

    cluster_score_dict = {}
    for cluster in clusters:
        best_solver = func_select_best_solver(yhat, cluster, db_instance, timeout)
        cluster_score = func_score_single_cluster(yhat, cluster, db_instance, timeout, best_solver)
        cluster_score_dict[cluster] = (best_solver, cluster_score)

    final_score = func_score_complete_clustering(yhat, clusters, db_instance, timeout, cluster_score_dict)

    return final_score

# -- func_select_best_solver(yhat, cluster, db_instance, timeout) ------------------------------------------------------


def f1_relative(yhat, cluster, db_instance, timeout):
    solver_dict = scoring.score_solvers_on_relative_runtime_cluster(yhat, cluster, db_instance, timeout)
    best_solver = ''
    best_score = 0
    for solver, solver_score in solver_dict.items():
        if solver_score >= best_score:
            best_solver = solver
            best_score = solver_score

    return best_solver


def f1_par2(yhat, cluster, db_instance, timeout):
    best_solver = ''
    best_score = 100000000000000
    for solver in db_instance.solver_f:
        current_score = f2_par2_cluster(yhat, cluster, db_instance, timeout)
        if current_score <= best_score:
            best_score = current_score
            best_solver = solver

    return best_solver


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

    return score / cluster_size

# -- func_score_complete_clustering(yhat, clusters, db_instance, timeout, cluster_score_dict) --------------------------


def f3_weigh_with_cluster_size(yhat, clusters, db_instance, timeout, cluster_score_dict):
    cluster_sizes = Counter(yhat)
    current_score = 0
    for cluster, (best_solver, cluster_score) in cluster_score_dict.items():
        current_score = current_score + cluster_score * cluster_sizes[cluster]

    final_score = current_score / len(yhat)
    return final_score

