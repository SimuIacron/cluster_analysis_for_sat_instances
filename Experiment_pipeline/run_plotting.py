from Experiment_pipeline.run_experiments import read_json


# gets the evaluations (or experiments) depending on the given settings_dict from the input_file
# the settings dict, needs to have the same structure as the dict stored in "settings" in the input file,
# however instead of values, for each settings list of settings is used, to enable the usage of "or"
# example: In the file the structure of the problem is given as {settings: {"scaling_algorithm": "SCALEMINUSPLUS1" }}
# then the settings dict needs to hve the structure: {"scaling_algorithm": ["SCALEMINUSPLUS1"] }, we can especially use
# {"scaling_algorithm": ["SCALEMINUSPLUS1", "SCALE01"] } to select evaluations with different scaling algorithm

# returns the list of evaluations that fit the dict and a diff which contains all keys, that where missing in the
# settings_dict, but where present in at least one selected evaluation
def collect_evaluation(input_file, settings_dict):
    data = read_json(input_file)[0]
    collected_experiment = []
    diff_set = set()
    for experiment in data:
        matchingValues = True
        for key, value in settings_dict.items():
            if experiment['settings'][key] not in value:
                matchingValues = False
                break

        if matchingValues:
            collected_experiment.append(experiment)
            diff = experiment['settings'].keys() - settings_dict.keys()
            diff_set = set.union(diff_set, diff)

    return collected_experiment, diff_set



