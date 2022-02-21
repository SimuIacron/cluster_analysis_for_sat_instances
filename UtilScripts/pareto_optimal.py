# this import structure is used, because some servers had problems installing oapackage
try:
    import oapackage
except ImportError:
    oapackage = None


# calculates the indices of pareto-optimal 2d-points
def get_pareto_indices(data, minimize_dim=None):
    length = len(data[0])
    pareto_data = []
    if minimize_dim is None:
        minimize_dim = [False] * length

    for i, dim in enumerate(data):
        assert len(dim) == length, 'data lists do not have the same length'
        if minimize_dim[i]:
            pareto_data.append([-x for x in dim])
        else:
            pareto_data.append(dim)

    pareto = oapackage.ParetoDoubleLong()
    for i in range(length):
        point = []
        for dim in pareto_data:
            point.append(dim[i])
        w = oapackage.doubleVector(tuple(point))
        pareto.addvalue(w, i)

    return pareto.allindices()


