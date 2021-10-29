from dash import dcc


def create_layout():
    return [
        dcc.Graph(
            id='clustering-graph-1',
            style={'height': 800},
        ),
        dcc.Graph(
            id='clustering-graph-2',
            style={'height': 800},
        ),
        dcc.Graph(
            id='clustering-graph-3',
            style={'height': 800},
        ),
    ]