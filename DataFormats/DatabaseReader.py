import os
import sys

from gbd_tool.gbd_api import GBD

from UtilScripts import util

# creates the Path to all relevant databases for the GBD Object
DB_PATH = os.environ["DBPATH"] + "meta.db" + os.pathsep + \
          os.environ["DBPATH"] + "base.db" + os.pathsep + \
          os.environ["DBPATH"] + "gate.db" + os.pathsep + \
          os.environ["DBPATH"] + "runtimes.db"

TIMEOUT = 5000

FEATURES_GATE = ['n_vars', 'n_gates', 'n_roots', 'n_none', 'n_generic', 'n_mono', 'n_and', 'n_or', 'n_triv', 'n_equiv',
                 'n_full', 'levels_mean', 'levels_variance', 'levels_min', 'levels_max', 'levels_entropy',
                 'levels_none_mean', 'levels_none_variance', 'levels_none_min', 'levels_none_max',
                 'levels_none_entropy',
                 'levels_generic_mean', 'levels_generic_variance', 'levels_generic_min', 'levels_generic_max',
                 'levels_generic_entropy', 'levels_mono_mean', 'levels_mono_variance', 'levels_mono_min',
                 'levels_mono_max',
                 'levels_mono_entropy', 'levels_and_mean', 'levels_and_variance', 'levels_and_min', 'levels_and_max',
                 'levels_and_entropy', 'levels_or_mean', 'levels_or_variance', 'levels_or_min', 'levels_or_max',
                 'levels_or_entropy', 'levels_triv_mean', 'levels_triv_variance', 'levels_triv_min', 'levels_triv_max',
                 'levels_triv_entropy', 'levels_equiv_mean', 'levels_equiv_variance', 'levels_equiv_min',
                 'levels_equiv_max',
                 'levels_equiv_entropy', 'levels_full_mean', 'levels_full_variance', 'levels_full_min',
                 'levels_full_max',
                 'levels_full_entropy', 'gate_features_runtime']

FEATURES_BASE = ['clauses', 'variables', 'clause_size_1', 'clause_size_2', 'clause_size_3', 'clause_size_4',
                 'clause_size_5', 'clause_size_6', 'clause_size_7', 'clause_size_8', 'clause_size_9', 'horn_clauses',
                 'inv_horn_clauses', 'positive_clauses', 'negative_clauses', 'horn_vars_mean', 'horn_vars_variance',
                 'horn_vars_min', 'horn_vars_max', 'horn_vars_entropy', 'inv_horn_vars_mean', 'inv_horn_vars_variance',
                 'inv_horn_vars_min', 'inv_horn_vars_max', 'inv_horn_vars_entropy', 'balance_clause_mean',
                 'balance_clause_variance', 'balance_clause_min', 'balance_clause_max', 'balance_clause_entropy',
                 'balance_vars_mean', 'balance_vars_variance', 'balance_vars_min', 'balance_vars_max',
                 'balance_vars_entropy', 'vcg_vdegrees_mean', 'vcg_vdegrees_variance', 'vcg_vdegrees_min',
                 'vcg_vdegrees_max', 'vcg_vdegrees_entropy', 'vcg_cdegrees_mean', 'vcg_cdegrees_variance',
                 'vcg_cdegrees_min', 'vcg_cdegrees_max', 'vcg_cdegrees_entropy', 'vg_degrees_mean',
                 'vg_degrees_variance', 'vg_degrees_min', 'vg_degrees_max', 'vg_degrees_entropy', 'cg_degrees_mean',
                 'cg_degrees_variance', 'cg_degrees_min', 'cg_degrees_max', 'cg_degrees_entropy',
                 'base_features_runtime']

# runtimes.db
FEATURES_SOLVER = ['cadical_elimfalse', 'cadical', 'cadical_pripro', 'cadical_stability', 'candy', 'glucose_chanseok',
                   'glucose', 'glucose_syrup', 'glucose_var_decay099', 'kissat', 'lingeling', 'march_nh', 'minisat',
                   'relaxed', 'yalsat']

FEATURES_FAMILY = ['family']
FEATURES_RESULT = ['result']


# removes keywords "empty", "memout", "timeout" and "failed" from the database and replaces them with
# with values depending on the feature and keyword
def remove_keywords_from_query_with_hash(query, features):

    empties = 0
    other_missing = 0

    for inst in query:
        for i in range(1, len(inst)):

            if inst[i] == "empty":
                inst[i] = -1
                empties = empties + 1
            elif inst[i] == "memout" or inst[i] is None or inst[i] == 'timeout' or inst[i] == 'failed':
                other_missing = other_missing + 1
                if features == FEATURES_SOLVER:
                    inst[i] = TIMEOUT
                else:
                    inst[i] = -sys.maxsize - 1
            else:
                try:
                    inst[i] = float(inst[i])

                # needed to let discrete values such as families intact
                except ValueError:
                    pass

    # print(empties)
    # print(other_missing)

    return query


# combines queries into one query for both the variant with and without hash
def combine_queries(queries, queries_without_hash):
    # combine the queries, to one
    # check if all queries have the same order, if not throw an error for now
    final_query_without_hash = []
    final_query = []
    for i in range(len(queries[0])):
        for j in range(len(queries)):
            if queries[0][i][0] != queries[j][i][0]:
                raise AssertionError()

    final_query_without_hash = []
    final_query = []
    for i in range(len(queries[0])):

        instance_without_hash = []
        instance = [queries[0][i][0]]
        for j in range(len(queries_without_hash)):
            instance = instance + queries_without_hash[j][i]
            instance_without_hash = instance_without_hash + queries_without_hash[j][i]
        final_query.append(instance)
        final_query_without_hash.append(instance_without_hash)

    return final_query, final_query_without_hash


# removes all instances based on the removed_indexes array, that marks instances to keep with 1 and instance to
# remove with 0
def remove_with_index_array(dataset, removed_indexes):
    result = []
    for i in range(len(removed_indexes)):
        if removed_indexes[i] == 1:
            result.append(dataset[i])

    return result


# generates an array that contains a 1 if the instances contains no missing entries, and 0 if it does
def generate_empty_array(dataset):
    # result = []
    removed_indexes = []

    for inst in dataset:
        if -1 not in inst:
            # result.append(inst)
            removed_indexes.append(1)
        else:
            removed_indexes.append(0)

    return removed_indexes


# reads the given features from the db and returns a query with and without the instance hashes
def read_from_db(features):
    with GBD(DB_PATH) as gbd:
        # split the features into blocks of no more than 64
        # otherwise the database can't handle the query
        split_features = util.chunks(features, 60)

        # do the query and create one query with, and one without the hash
        queries = []
        queries_without_hash = []

        # hash_list = []

        print(features)

        for sub_features in split_features:
            query = gbd.query_search("local like %sat202%", [], sub_features)

            # for inst in query:
            #     for i in inst:
            #         try:
            #             if float(i) < 0:
            #                 hash_list.append(inst[0])
            #         except ValueError:
            #             pass
            #
            # print(hash_list)

            query = [list(i) for i in query]
            query = remove_keywords_from_query_with_hash(query, sub_features)
            query_without_hash = [el[1:] for el in query]
            queries.append(query)
            queries_without_hash.append(query_without_hash)

        return combine_queries(queries, queries_without_hash)


# -------------------------------------------------------
# all functions below return a tuple of the requested features with
# and without hash
# it also appends the list of feature names
# -------------------------------------------------------
def read_family_from_db():
    return read_from_db(FEATURES_FAMILY), FEATURES_FAMILY


def read_solver_from_db():
    return read_from_db(FEATURES_SOLVER), FEATURES_SOLVER


def read_gate_from_db():
    return read_from_db(FEATURES_GATE), FEATURES_GATE


def read_base_from_db():
    return read_from_db(FEATURES_BASE), FEATURES_BASE


def read_result_from_db():
    return read_from_db(FEATURES_RESULT), FEATURES_RESULT
