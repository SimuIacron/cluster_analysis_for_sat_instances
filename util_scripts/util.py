# splits list lst into chunks of n
def chunks(lst, n):
    n = max(1, n)
    return [lst[i:i+n] for i in range(0, len(lst), n)]


# scales given array values to a scale of 0 to 1
# example: [3,5,7] --> [0, 0.5, 1]
def scale_array_to_01(array):
    max_v = max(array)
    min_v = min(array)
    if max_v != 0:
        scaled_array = [((value - min_v) / (max_v-min_v)) for value in array]
    else:
        scaled_array = [0] * len(array)

    return scaled_array


# scales given array values to a scale of 0 to 1
# example: [3,5,7] --> [-1, 0, 1]
def scale_array_to_minus_plus_1(array):
    max_v = max(array)
    min_v = min(array)

    center = (max_v + min_v)/2
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

