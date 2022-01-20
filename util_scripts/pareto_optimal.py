import oapackage


# calculates the indices of pareto-optimal 2d-points
def get_pareto_indices_2d(x_data, y_data, minimize_x=False, minimize_y=False):
    assert len(x_data) == len(y_data), 'data lists do not have the same length'

    if minimize_x:
        x_ = [-x for x in x_data]
    else:
        x_ = x_data
    if minimize_y:
        y_ = [-y for y in y_data]
    else:
        y_ = y_data

    pareto = oapackage.ParetoDoubleLong()
    for idx, (x, y) in enumerate(zip(x_, y_)):
        w = oapackage.doubleVector((x, y))
        pareto.addvalue(w, idx)

    return pareto.allindices()


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

