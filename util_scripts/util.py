# splits list lst into chunks of n
import itertools
import random

from DataFormats import DatabaseReader


def chunks(lst, n):
    n = max(1, n)
    return [lst[i:i + n] for i in range(0, len(lst), n)]


# scales given array values to a scale of 0 to 1
# example: [3,5,7] --> [0, 0.5, 1]
def scale_array_to_01(array):
    max_v = max(array)
    min_v = min(array)
    if max_v-min_v != 0:
        scaled_array = [((value - min_v) / (max_v - min_v)) for value in array]
    else:
        scaled_array = [0] * len(array)

    return scaled_array


# scales given array values to a scale of 0 to 1
# example: [3,5,7] --> [-1, 0, 1]
def scale_array_to_minus_plus_1(array):
    max_v = max(array)
    min_v = min(array)

    center = (max_v + min_v) / 2
    divisor = max_v - center

    if divisor != 0:
        scaled_array = [((value - center) / divisor) for value in array]
    else:
        scaled_array = [0] * len(array)

    return scaled_array


# rotates given nested list
# example: [[1,2], [3,4], [5,6]] --> [[1,3,5], [2,4,6]]
def rotateNestedLists(nested_list):
    return list(map(list, zip(*nested_list)))


# selected the k smallest values in an array an sets them to 1
# all other values get set to 0
# example k = 2 [2,5,1,7,3,9] --> [1,0,1,0,0,0]
def select_k_best_mins(array, k):
    sorted_array = sorted(array)
    k_best = sorted_array[:k]
    output = [0] * len(array)
    for item in k_best:
        output[array.index(item)] = 1
    return output


def flatten(t):
    final_list = []
    for sublist in t:
        for item in sublist:
            final_list.append(item)

    return final_list


# Replaces every nth occurrence of letter in text with line break
# text: The inpout text to add breaks to
# letter_to_break_at: The letter that is replaced by the break
# replace_n: Every nth letter gets replaced by a break
def add_line_breaks_to_text(text, letter_to_break_at, replace_n):
    text_string = ''
    n_counter = 0
    for letter in text:
        if letter == letter_to_break_at:
            n_counter = n_counter + 1
            if replace_n == n_counter:
                n_counter = 0
                text_string = text_string + ',<br>'
            else:
                text_string = text_string + ','
        else:
            text_string = text_string + letter

    return text_string


def random_color():
    return random.uniform(0, 1), random.uniform(0, 1), random.uniform(0, 1)


def get_combinations_of_databases(use_base=True, use_gate=True, use_solver=True):
    temp_solver_features = DatabaseReader.FEATURES_SOLVER.copy()
    temp_solver_features.pop(14)
    temp_solver_features.pop(7)

    input_dbs = []
    if use_base:
        input_dbs.append(DatabaseReader.FEATURES_BASE)
    if use_gate:
        input_dbs.append(DatabaseReader.FEATURES_GATE)
    if use_solver:
        input_dbs.append(temp_solver_features)

    output = sum([list(map(list, itertools.combinations(input_dbs, i))) for i in range(len(input_dbs) + 1)], [])
    output_merged = []
    for combination in output:
        comb = []
        for elem in combination:
            comb = comb + elem
        output_merged.append(comb)

    features = []
    for feature_vector in input_dbs:
        features = features + feature_vector

    return output_merged[1:], features
