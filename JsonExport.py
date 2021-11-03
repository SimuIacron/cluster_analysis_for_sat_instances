import json
from datetime import datetime
import os

from DataFormats.InputData import InputDataFeatureSelection, InputDataCluster, InputDataScaling


def export_json(dataset, cluster_params: InputDataCluster, feature_selection_params: InputDataFeatureSelection,
                scaling_params: InputDataScaling):

    misc_dict = {'dataset_selection': dataset}

    cluster_dict = {key: value for key, value in cluster_params.__dict__.items() if
                    not key.startswith('__') and not callable(key)}
    feature_selection_dict = {key: value for key, value in feature_selection_params.__dict__.items() if
                              not key.startswith('__') and not callable(key)}
    scaling_dict = {key: value for key, value in scaling_params.__dict__.items() if
                    not key.startswith('__') and not callable(key)}

    dict_params = {'misc': misc_dict, 'cluster_params': cluster_dict,
                   'feature_selection_params': feature_selection_dict, 'scaling_dict': scaling_dict}

    dict_final = {'params': dict_params}

    now = datetime.now()
    dt_string = now.strftime("%y-%m-%d_%H-%M-%S")

    with open(os.environ['JSONPATH'] + dt_string, 'w') as outfile:
        json.dump(dict_final, outfile)


def convert_bytes_to_dict(data):
    return json.loads(data.decode('utf-8'))
