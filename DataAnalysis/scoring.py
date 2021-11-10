import DatabaseReader
import util
from DataFormats.DbInstance import DbInstance
from sklearn.metrics import adjusted_mutual_info_score, adjusted_rand_score, completeness_score, fowlkes_mallows_score, \
    homogeneity_score, mutual_info_score, normalized_mutual_info_score, rand_score, v_measure_score
from collections import Counter


def get_k_best_solvers_for_instance(instance_solver_times, k):
    solver_features = DatabaseReader.FEATURES_SOLVER
    sorted_features = [x for _, x in sorted(zip(instance_solver_times, solver_features))]
    return sorted_features[:k]


def get_best_solver_int(db_instance: DbInstance):
    solver_features = DatabaseReader.FEATURES_SOLVER
    final = []
    for inst in db_instance.solver_wh:
        sorted_features = [x for _, x in sorted(zip(inst, solver_features))]
        final.append(solver_features.index(sorted_features[0]))

    return final



def score_solvers_on_rank_cluster(yhat, cluster_idx, db_instance: DbInstance, ranks, factors):

    rank_count = [[] for dummy in range(len(ranks))]
    cluster_amount = 0
    for i in range(len(yhat)):
        if yhat[i] == cluster_idx:
            cluster_amount = cluster_amount + 1
            for j in range(len(db_instance.solver_wh[i])):
                counter = 0
                while counter < len(ranks):
                    if db_instance.solver_wh[i][j] <= ranks[counter]:
                        rank_count[counter].append(db_instance.solver_f[j])
                        break

                    counter = counter + 1

    solver_dict = {}
    for i in range(len(rank_count)):
        count = Counter(rank_count[i])
        for key, value in count.items():
            if key not in solver_dict:
                solver_dict[key] = 0

            solver_dict[key] = solver_dict[key] + factors[i] * value

    for key, value in solver_dict.items():
        solver_dict[key] = value / (len(db_instance.solver_wh[0]) * cluster_amount)
    return solver_dict


def score_solvers_squared_relativ(yhat, db_instance: DbInstance, k):
    cluster_dict = {}
    for i in range(len(yhat)):
        cluster = yhat[i]
        if cluster not in cluster_dict:
            cluster_dict[cluster] = []

        cluster_dict[cluster].append(get_k_best_solvers_for_instance(db_instance.solver_wh[i], k))

    score_dict = {}
    for key, value in cluster_dict.items():
        flatten_solver = util.flatten(value)
        total_amount = len(flatten_solver)
        count = Counter(flatten_solver)
        score = 0
        for algo, amount in count.items():
            score = score + (amount/total_amount) * (amount/total_amount)
        score_dict[key] = score * k

    return score_dict



def score_solver_similarity_in_cluster_occurence(yhat, db_instance:DbInstance, k):
    cluster_dict = {}
    for i in range(len(yhat)):
        cluster = yhat[i]
        if cluster not in cluster_dict:
            cluster_dict[cluster] = []

        cluster_dict[cluster].append(get_k_best_solvers_for_instance(db_instance.solver_wh[i], k))

    score_dict = {}
    for key, value in cluster_dict.items():
        flatten_solver = util.flatten(value)
        amount = len(list(dict.fromkeys(flatten_solver)))
        score = amount / k
        score_dict[key] = score

    pass



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


def score_cluster_family(yhat, db_instance: DbInstance):
    scoring_list = []
    family_int = convert_families_to_int(db_instance.family_wh)

    scoring_list.append(('adjusted mutual information score', adjusted_mutual_info_score(family_int, yhat)))
    scoring_list.append(('adjusted rand score', adjusted_rand_score(family_int, yhat)))
    scoring_list.append(('completeness score', completeness_score(family_int, yhat)))
    scoring_list.append(('fowlkes mallows score', fowlkes_mallows_score(family_int, yhat)))
    scoring_list.append(('homogeneity score', homogeneity_score(family_int, yhat)))
    scoring_list.append(('mutual info score', mutual_info_score(family_int, yhat)))
    scoring_list.append(('normalized mutual info score', normalized_mutual_info_score(family_int, yhat)))
    scoring_list.append(('rand score', rand_score(family_int, yhat)))
    scoring_list.append(('v measure score', v_measure_score(family_int, yhat)))
    print(scoring_list)

    score_dict_time = score_solvers_squared_relativ(yhat, db_instance, 3)

    return scoring_list, score_dict_time
