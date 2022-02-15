from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import StandardScaler
from UtilScripts import util
from DataFormats import DatabaseReader

SCALINGALGORITHMS = [
    ('Scale to [-1,1]', 'SCALEMINUSPLUS1'),
    ('Standard Scaler', 'STANDARDSCALER'),
    ('Scale to [0,1]', 'SCALE01'),
    ('No Scaling', 'NONE')
]

SCALINGTECHNIQUES = [
    ('Scale each feature to [-1, 1]', 'NORMALSCALE'),
    ('Scale instance features per feature to [-1, 1] and scale the solver times per instance from [-1, 1]', 'TIMESCALE'),
    ('Scale instance features per feature to [-1, 1] mark best k solver times in instance with 1, rest 0', 'TIMESELECTBEST')
]


def scaling(data, features, params_dict):
    algorithm = params_dict['scaling_algorithm']

    # it is assumed that the solver_time features are the last ones in the data set
    solver_start_index = -1
    if DatabaseReader.FEATURES_SOLVER[0] in features:
        solver_start_index = features.index(DatabaseReader.FEATURES_SOLVER[0])

    if algorithm == 'NORMALISATION':
        scaler = Normalizer()
        data_rot = util.rotateNestedLists(data)
        return util.rotateNestedLists(scaler.transform(data_rot))

    if algorithm == "STANDARDSCALER":
        scaler = StandardScaler()
        scaler.fit(data)
        return scaler.transform(data)
    elif algorithm == "SCALEMINUSPLUS1":
        # scale the instance features (gate and base) per feature from [-1,1] and
        # scale the solver time per instance form [-1, 1]
        if params_dict['scaling_technique'] == 'TIMESCALE' and solver_start_index != -1:
            scaled_data = scale_only_instances_features(solver_start_index, data)

            finished = []
            for instance in scaled_data:
                instance_features = instance[:solver_start_index]
                solver_times = instance[solver_start_index:]
                new_instance = instance_features + util.scale_array_to_minus_plus_1(solver_times)
                finished.append(new_instance)
            return finished

        # scale the instance features (gate and base) per feature form [-1,1] and
        # set the k best solvers for each instance to 1 and all other to 0
        elif params_dict['scaling_technique'] == 'TIMESELECTBEST' and solver_start_index != -1:
            scaled_data = scale_only_instances_features(solver_start_index, data)

            finished = []
            for instance in scaled_data:
                instance_features = instance[:solver_start_index]
                solver_times = instance[solver_start_index:]
                new_instance = instance_features + util.select_k_best_mins(solver_times, params_dict['scaling_k_best'])
                finished.append(new_instance)
            return finished

        # scale all features (base, gate, solver time) per feature form [-1,1]
        else: # params_dict['scaling_technique'] == 'NORMALSCALE'
            return util.rotateNestedLists([util.scale_array_to_minus_plus_1(feature) for feature in
                                           util.rotateNestedLists(data)])
    elif algorithm == "SCALE01":
        return util.rotateNestedLists([util.scale_array_to_01(feature) for feature in
                                       util.rotateNestedLists(data)])
    else:  # algorithm == 'NONE'
        return data


# scales only the instance features (base and gate) from [-1,1] per feature
# solver_start_index: Gives the index where the instance features end and the solver time features start
# data: The data to scale
def scale_only_instances_features(solver_start_index, data):
    rot_data = util.rotateNestedLists(data)
    rot_scaled_data = []
    for i in range(len(rot_data)):
        if i < solver_start_index:
            rot_scaled_data.append(util.scale_array_to_minus_plus_1(rot_data[i]))
        else:
            rot_scaled_data.append(rot_data[i])

    return util.rotateNestedLists(rot_scaled_data)
