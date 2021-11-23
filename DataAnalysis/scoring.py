from DataAnalysis import scoring_util, scoring_modular
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
        cluster_score = scoring_modular.f2_par2_cluster(yhat, cluster, db_instance, timeout, best_solver)

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
        best_score = scoring_modular.f2_par2_cluster(yhat, cluster, db_instance, timeout, db_instance.solver_f[0])
        for solver in db_instance.solver_f[1:]:
            current_score = scoring_modular.f2_par2_cluster(yhat, cluster, db_instance, timeout, solver)
            if current_score < best_score:
                best_score = current_score
                best_solver = solver

        cluster_scores.append(best_score)
        cluster_algo.append(best_solver)
        score = score + best_score * cluster_sizes[cluster]

    final_score = score / len(yhat)
    return final_score, cluster_scores, cluster_algo





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


# calculates the van dongen criterion normalized as described in the paper
# Wu (2009) “Adapting the right measures for k-means clustering”
def van_dongen_normalized(labels_pred, labels_true):
    n = len(labels_pred)
    c = scoring_util.create_contingency_matrix(labels_pred, labels_true)
    c_t = scoring_util.create_contingency_matrix(labels_true, labels_pred)

    sum1 = 0
    for i in range(len(set(labels_pred))):
        sum1 = sum1 + max(c[i])

    sum2 = 0
    for j in range(len(set(labels_true))):
        sum2 = sum2 + max(c_t[j])

    vd = (2 * n - sum1 - sum2) / (2 * n - max(scoring_util.create_contingency_row(labels_pred)) - max(scoring_util.create_contingency_row(labels_true)))
    return vd


# collection of scoring algorithms given by sklearn
def score_cluster_family(yhat, db_instance: DbInstance):
    scoring_list = []
    family_int = scoring_util.convert_families_to_int(db_instance.family_wh)

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
