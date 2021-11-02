from sklearn.preprocessing import StandardScaler

import util
from DataFormats.InputData import InputDataScaling

SCALINGALGORITHMS = [
    ('Scale to [-1,1]', 'SCALEMINUSPLUS1'),
    ('Standard Scaler', 'STANDARDSCALER'),
    ('Scale to [0,1]', 'SCALE01'),
    ('No Scaling', 'NONE')
]


def scaling(data, params: InputDataScaling):
    algorithm = params.scaling_algorithm

    if algorithm == "STANDARDSCALER":
        scaler = StandardScaler()
        scaler.fit(data)
        return scaler.transform(data)
    elif algorithm == "SCALEMINUSPLUS1":
        return util.rotateNestedLists([util.scale_array_to_minus_plus_1(feature) for feature in
                                       util.rotateNestedLists(data)])
    elif algorithm == "SCALE01":
        return util.rotateNestedLists([util.scale_array_to_01(feature) for feature in
                                       util.rotateNestedLists(data)])
    else:  # algorithm == 'NONE'
        return data
