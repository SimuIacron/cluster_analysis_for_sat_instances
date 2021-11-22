import DatabaseReader
import util
from DataFormats.DbInstance import DbInstance
from sklearn.metrics import adjusted_mutual_info_score, adjusted_rand_score, completeness_score, fowlkes_mallows_score, \
    homogeneity_score, mutual_info_score, normalized_mutual_info_score, rand_score, v_measure_score
from collections import Counter
from numpy import max


# calculates par2 score of the best solvers of relative runtime of each cluster for the whole clustering
# clusters: A list of cluster indexes
# the clustering given by an array of cluster indexes so that each instances gets mapped to one cluster
# db_instance: Stores all data form the db files
# the timeout value
def score_clustering_par2_relative(clusters, yhat, db_instance:DbInstance, timeout):
    score = 0
    cluster_scores = []
    cluster_algo = []
    cluster_sizes = Counter(yhat)
    # iterate through each cluster
    for cluster in clusters:

        # get the solver scores by the relative runtime and find the solver with the best score
        solver_dict = score_solvers_on_relative_runtime_cluster(yhat, cluster, db_instance, timeout)
        best_solver = ''
        best_score = 0
        for solver, solver_score in solver_dict.items():
            if solver_score >= best_score:
                best_solver = solver
                best_score = solver_score

        # calculate the score of the best solver with the par2 score
        cluster_score = par2_cluster(best_solver, cluster, yhat, db_instance, timeout)

        cluster_scores.append(cluster_score)
        cluster_algo.append(best_solver)
        score = score + cluster_score * cluster_sizes[cluster]

    final_score = score / len(yhat)
    return final_score, cluster_scores, cluster_algo


# calculates par2 score of the best solvers of par2 score of each cluster for the whole clustering
# clusters: A list of cluster indexes
# the clustering given by an array of cluster indexes so that each instances gets mapped to one cluster
# db_instance: Stores all data form the db files
# the timeout value
def score_clustering_par2(clusters, yhat, db_instance:DbInstance, timeout):
    score = 0
    cluster_scores = []
    cluster_algo = []
    cluster_sizes = Counter(yhat)
    # iterate through each cluster
    for cluster in clusters:
        best_solver = db_instance.solver_f[0]
        best_score = par2_cluster(db_instance.solver_f[0], cluster, yhat, db_instance, timeout)
        for solver in db_instance.solver_f[1:]:
            current_score = par2_cluster(solver, cluster, yhat, db_instance, timeout)
            if current_score < best_score:
                best_score = current_score
                best_solver = solver

        cluster_scores.append(best_score)
        cluster_algo.append(best_solver)
        score = score + best_score * cluster_sizes[cluster]

    final_score = score / len(yhat)
    return final_score, cluster_scores, cluster_algo


# calculates the par2 score for a given cluster and solver
def par2_cluster(solver, cluster_idx, yhat, db_instance: DbInstance, timeout):
    solver_idx = db_instance.solver_f.index(solver)
    cluster_size = 0
    score = 0
    for idx, inst in enumerate(db_instance.solver_wh):
        if cluster_idx == yhat[idx]:
            cluster_size = cluster_size + 1
            running_time = inst[solver_idx]
            if running_time >= timeout:
                running_time = timeout * 2
            score = score + running_time

    return score / cluster_size


# Calculates the following metric for the cluster given with cluster_idx:
# 1. calculate the min running time over all cluster instances and solvers
# 2. Divide all running times by the min running time to get relative running times
# 3. Sort the relative running times into ranks, by factors 1,2,3, etc...
# 4. Weigh timeouts with times 2 when sorting into ranks
# 5. calculate score of each solver in the cluster by using the amounts of the solver in each cluster multiplied by the
# multiplier of the rank
# 6. normalize the scores
# 7. return dict of each solver with score
def score_solvers_on_relative_runtime_cluster(yhat, cluster_idx, db_instance: DbInstance, timeout):

    # get instances of cluster and min running time
    cluster_insts = []
    min_running_time = timeout
    for idx, inst in enumerate(db_instance.solver_wh):
        if cluster_idx == yhat[idx]:
            cluster_insts.append(inst)
            current_min = min(inst)
            if current_min < min_running_time and current_min != 0:
                min_running_time = current_min

    # make sure min running time is not zero
    # if min_running_time == 0:
    #    min_running_time = 0.01

    # sort solvers into ranks
    rank_amount = 5
    factor_divider = 1
    ranks = [[] for _ in range(rank_amount)]
    for i, inst in enumerate(cluster_insts):
        for j, item in enumerate(inst):
            if cluster_insts[i][j] == timeout:
                ranks[-1].append(db_instance.solver_f[j])
            else:

                factor = cluster_insts[i][j] / min_running_time
                factor = factor / factor_divider
                if int(factor)-1 < rank_amount:
                    ranks[int(factor)-1].append(db_instance.solver_f[j])
                else:
                    ranks[-2].append(db_instance.solver_f[j])

    # create dict of every solver with score
    solver_dict = {}
    for i in range(len(ranks)):
        count = Counter(ranks[i])
        for key, value in count.items():
            if key not in solver_dict:
                solver_dict[key] = 0

            # score calculation            use the rank index as factor   use the rank_amount*amount of instances to
            # normalize
            solver_dict[key] = solver_dict[key] + ((rank_amount-i-1) * value) / (rank_amount * len(cluster_insts))

    return solver_dict


# gets for the given instance (only an array of solver times of the instance)
# the best k solvers, which get determined by the shortest running time
def get_k_best_solvers_for_instance(instance_solver_times, k):
    solver_features = DatabaseReader.FEATURES_SOLVER
    sorted_features = [x for _, x in sorted(zip(instance_solver_times, solver_features))]
    return sorted_features[:k]


# returns a clustering which uses the best solvers as each cluster
# this means every instance which have the same best solver
# are considered to be in a cluster
def get_best_solver_int(db_instance: DbInstance):
    solver_features = DatabaseReader.FEATURES_SOLVER
    final = []
    for inst in db_instance.solver_wh:
        sorted_features = [x for _, x in sorted(zip(inst, solver_features))]
        final.append(solver_features.index(sorted_features[0]))

    return final


# calculates the score of a cluster, by giving each solver of instances a value through
# linear interpolation of [0,timeout] --> [1,0]
# then adding the values of all solvers and
# returning the score of the highest solver
def score_solvers_on_linear_rank_cluster(yhat, cluster_idx, db_instance: DbInstance, timeout):
    cluster_amount = 0
    solver_dict = {}
    for i in range(len(yhat)):
        if yhat[i] == cluster_idx:
            cluster_amount = cluster_amount + 1
            for j in range(len(db_instance.solver_wh[0])):
                if db_instance.solver_f[j] not in solver_dict:
                    solver_dict[db_instance.solver_f[j]] = 0
                add_value = ((db_instance.solver_wh[i][j] / timeout) - 1) * (-1)
                solver_dict[db_instance.solver_f[j]] = solver_dict[db_instance.solver_f[j]] + add_value

    best_solver = ''
    best_solver_score = 0
    for key, value in solver_dict.items():
        if value > best_solver_score:
            best_solver = key
            best_solver_score = value

    return best_solver, best_solver_score / cluster_amount


# scores a cluster by mapping each solver by it's time on the instances to a rank (given by the rank array)
# then calculates the score of each solver by the factors of each ranks
# return the best solver score
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

    best_solver = ''
    best_solver_score = 0
    for key, value in solver_dict.items():
        score = value / (factors[0] * cluster_amount)
        if score > best_solver_score:
            best_solver = key
            best_solver_score = score
    return best_solver, best_solver_score


# returns a cluster by families
# this means all instances that have the same family are considered to be in the same cluster
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


def convert_sat_unsat_to_int(instance_result):
    instance_result_int = []
    for item in instance_result:
        if item[0] == 'sat':
            instance_result_int.append(0)
        else:
            instance_result_int.append(1)

    return instance_result_int



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


# calculates the van dongen criterion normalized as described in the paper
# Wu (2009) “Adapting the right measures for k-means clustering”
def van_dongen_normalized(labels_pred, labels_true):
    n = len(labels_pred)
    c = create_contingency_matrix(labels_pred, labels_true)
    c_t = create_contingency_matrix(labels_true, labels_pred)

    sum1 = 0
    for i in range(len(set(labels_pred))):
        sum1 = sum1 + max(c[i])

    sum2 = 0
    for j in range(len(set(labels_true))):
        sum2 = sum2 + max(c_t[j])

    vd = (2 * n - sum1 - sum2) / (2 * n - max(create_contingency_row(labels_pred)) - max(create_contingency_row(labels_true)))
    return vd


# collection of scoring algorithms given by sklearn
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

    return scoring_list
