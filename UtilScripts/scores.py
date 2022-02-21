import numpy as np

from DataFormats.DbInstance import DbInstance
from UtilScripts.util import rotateNestedLists


def par2(solver_name, db_instance: DbInstance, instance_index_list, timeout):
    solver_index = db_instance.solver_f.index(solver_name)
    solver = rotateNestedLists(db_instance.solver_wh)[solver_index]

    runtimes_of_instances = []
    for instance in instance_index_list:
        runtimes_of_instances.append(solver[instance])

    par2_ = 0
    for runtime in runtimes_of_instances:
        if runtime >= timeout:
            par2_ = par2_ + timeout * 2
        else:
            par2_ = par2_ + runtime

    return par2_ / len(runtimes_of_instances)


# def spar2_old(solver_name, db_instance: DbInstance, instance_index_list, timeout):
#     solver_index = db_instance.solver_f.index(solver_name)
#     solver = rotateNestedLists(db_instance.solver_wh)[solver_index]
#
#     runtimes_of_instances = []
#     for instance in instance_index_list:
#         runtimes_of_instances.append(solver[instance])
#
#     # std_ = std(runtimes_of_instances)
#     par2_ = par2(solver_name, db_instance, instance_index_list, timeout)
#
#     std_ = 0
#     for runtime in runtimes_of_instances:
#         if runtime >= timeout:
#             runtime = timeout * 2
#         std_ = std_ + ((runtime - par2_) * (runtime - par2_))
#     std_ = np.sqrt(std_ / len(runtimes_of_instances))
#
#     return par2_ + std_


def spar2(solver_name, db_instance: DbInstance, instance_index_list, timeout):
    solver_index = db_instance.solver_f.index(solver_name)
    solver = rotateNestedLists(db_instance.solver_wh)[solver_index]

    par2_ = par2(solver_name, db_instance, instance_index_list, timeout)

    runtimes_of_instances = []
    timeouts_of_instances = []
    for instance in instance_index_list:
        runtime = solver[instance]
        if runtime >= timeout:
            timeouts_of_instances.append(runtime)
        else:
            runtimes_of_instances.append(solver[instance])

    mad_ = 0
    if len(runtimes_of_instances) != 0:
        mean_ = np.mean(runtimes_of_instances)
        for runtime in runtimes_of_instances:
            mad_ = mad_ + np.abs(mean_ - runtime)
        mad_ = mad_ / len(runtimes_of_instances)

    penalized_std = (len(runtimes_of_instances) * mad_
                     + len(timeouts_of_instances * 2) * timeout) / len(instance_index_list)

    return penalized_std


def clustering_best(clustering, db_instances: DbInstance, scoring_function, timeout):
    clusters = {}

    for i, cluster_value in enumerate(clustering):
        if cluster_value not in clusters:
            clusters[cluster_value] = []
        clusters[cluster_value].append(i)

    c_score = 0
    for key, item in clusters.items():
        min_score = timeout * 1000
        for solver in db_instances.solver_f:
            solver_score = scoring_function(solver, db_instances, item, timeout)
            if solver_score < min_score:
                min_score = solver_score

        c_score = c_score + len(item) * min_score

    return c_score / len(clustering)


def virtual_best(db_instances: DbInstance, scoring_function, timeout):
    instance_number = len(db_instances.solver_wh)

    v_score = 0
    for i in range(0, instance_number):
        min_score = timeout * 1000
        for solver in db_instances.solver_f:
            solver_score = scoring_function(solver, db_instances, [i], timeout)
            if solver_score < min_score:
                min_score = solver_score

        v_score = v_score + min_score

    return v_score / instance_number


def single_best(db_instances: DbInstance, scoring_function, timeout):
    best_solver = ''
    min_score = timeout * 1000
    for solver in db_instances.solver_f:
        solver_score = scoring_function(solver, db_instances, range(0, len(db_instances.solver_wh)), timeout)
        if solver_score < min_score:
            min_score = solver_score
            best_solver = solver

    return min_score, best_solver


def cbs(clustering, db_instances: DbInstance, timeout):
    return clustering_best(clustering, db_instances, par2, timeout)


def cbss(clustering, db_instances: DbInstance, timeout):
    return clustering_best(clustering, db_instances, spar2, timeout)


def sbs(db_instances: DbInstance, timeout):
    return single_best(db_instances, par2, timeout)


def sbss(db_instances: DbInstance, timeout):
    return single_best(db_instances, spar2, timeout)


def vbs(db_instances: DbInstance, timeout):
    return virtual_best(db_instances, par2, timeout)


def vbss(db_instances: DbInstance, timeout):
    return virtual_best(db_instances, spar2, timeout)
