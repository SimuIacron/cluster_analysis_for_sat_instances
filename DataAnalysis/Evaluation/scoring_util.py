# returns a cluster by families
# this means all instances that have the same family are considered to be in the same cluster
from util import DatabaseReader
from DataFormats.DbInstance import DbInstance


def convert_families_to_int(instance_family):
    counter = 0
    family_dict = {}
    instance_family_int = []
    for item in instance_family:
        if item[0] not in family_dict:
            family_dict[item[0]] = counter
            counter = counter + 1

        instance_family_int.append(family_dict[item[0]])

    return instance_family_int


def convert_un_sat_to_int(instance_result):
    instance_result_int = []
    for item in instance_result:
        if item[0] == 'sat':
            instance_result_int.append(0)
        else:
            instance_result_int.append(1)

    return instance_result_int


# returns a clustering which uses the best solvers as each cluster
# this means every instance which have the same best solver
# are considered to be in a cluster
def convert_best_solver_int(db_instance: DbInstance):
    solver_features = DatabaseReader.FEATURES_SOLVER
    final = []
    for inst in db_instance.solver_wh:
        sorted_features = [x for _, x in sorted(zip(inst, solver_features))]
        final.append(solver_features.index(sorted_features[0]))

    return final


# gets for the given instance (only an array of solver times of the instance)
# the best k solvers, which get determined by the shortest running time
def get_k_best_solvers_for_instance(instance_solver_times, k):
    solver_features = DatabaseReader.FEATURES_SOLVER
    sorted_features = [x for _, x in sorted(zip(instance_solver_times, solver_features))]
    return sorted_features[:k]


# creates a contingency matrix of the two given clusterings
def create_contingency_matrix(labels_pred, labels_true):
    labels = list(zip(labels_pred, labels_true))
    pred_len = len(set(labels_pred))
    true_len = len(set(labels_true))
    contingency_matrix = [[0] * true_len for dummy in range(pred_len)]

    for (i, j) in labels:
        contingency_matrix[i][j] = contingency_matrix[i][j] + 1

    return contingency_matrix


# creates one sum row of the contingency tables of a clustering
# this is just a list which counts how many of each cluster occur in the given clustering
# for example [0,0,1,2,2,0] --> [3, 1, 2]
def create_contingency_row(labels):
    row_len = len(set(labels))
    row = [0] * row_len

    for i in labels:
        row[i] = row[i] + 1

    return row

