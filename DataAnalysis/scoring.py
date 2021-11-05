from DataFormats.DbInstance import DbInstance
from sklearn.metrics import adjusted_mutual_info_score, adjusted_rand_score, completeness_score, fowlkes_mallows_score, \
    homogeneity_score, mutual_info_score, normalized_mutual_info_score, rand_score, v_measure_score


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

    scoring_list.append(('adjusted mutual information score', adjusted_mutual_info_score(yhat, family_int)))
    scoring_list.append(('adjusted rand score', adjusted_rand_score(yhat, family_int)))
    scoring_list.append(('completeness score', completeness_score(yhat, family_int)))
    scoring_list.append(('fowlkes mallows score', fowlkes_mallows_score(yhat, family_int)))
    scoring_list.append(('homogeneity score', homogeneity_score(yhat, family_int)))
    scoring_list.append(('mutual info score', mutual_info_score(yhat, family_int)))
    scoring_list.append(('normalized mutual info score', normalized_mutual_info_score(yhat, family_int)))
    scoring_list.append(('rand score', rand_score(yhat, family_int)))
    scoring_list.append(('v measure score', v_measure_score(yhat, family_int)))
    print(scoring_list)

    return scoring_list
