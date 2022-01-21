import numpy as np

from DataFormats.DbInstance import DbInstance
from util_scripts import util


def calculate_stochastic_value_for_dataset(db_instance: DbInstance):
    base_rot = util.rotateNestedLists(db_instance.base_wh)
    base_mean = [np.mean(item) for item in base_rot]
    base_variance = [np.var(item) for item in base_rot]
    base_std = [np.std(item) for item in base_rot]
    base_min = [np.min(item) for item in base_rot]
    base_max = [np.max(item) for item in base_rot]

    gate_rot = util.rotateNestedLists(db_instance.gate_wh)
    gate_mean = [np.mean(item) for item in gate_rot]
    gate_variance = [np.var(item) for item in gate_rot]
    gate_std = [np.std(item) for item in gate_rot]
    gate_min = [np.min(item) for item in gate_rot]
    gate_max = [np.max(item) for item in gate_rot]

    runtimes_rot = util.rotateNestedLists(db_instance.solver_wh)
    runtimes_mean = [np.mean(item) for item in runtimes_rot]
    runtimes_variance = [np.var(item) for item in runtimes_rot]
    runtimes_std = [np.std(item) for item in runtimes_rot]
    runtimes_min = [np.min(item) for item in runtimes_rot]
    runtimes_max = [np.max(item) for item in runtimes_rot]

    return {
        'base_mean': base_mean,
        'base_variance': base_variance,
        'base_std': base_std,
        'base_min': base_min,
        'base_max': base_max,
        'gate_mean': gate_mean,
        'gate_variance': gate_variance,
        'gate_std': gate_std,
        'gate_min': gate_min,
        'gate_max': gate_max,
        'runtimes_mean': runtimes_mean,
        'runtimes_variance': runtimes_variance,
        'runtimes_std': runtimes_std,
        'runtimes_min': runtimes_min,
        'runtimes_max': runtimes_max
    }

