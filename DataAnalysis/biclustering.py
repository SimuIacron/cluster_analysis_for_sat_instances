import numpy as np
from sklearn.cluster import SpectralCoclustering, SpectralBiclustering



def bicluster(instance_list, algorithm="SPECTRALCO"):

    if algorithm == "SPECTRALBI":
        model = SpectralBiclustering(n_clusters=10, random_state=0)
    else:
        model = SpectralCoclustering(n_clusters=10, random_state=0)

    model.fit(instance_list)

    fit_data = instance_list[np.argsort(model.row_labels_)]
    fit_data = fit_data[:, np.argsort(model.column_labels_)]

    return fit_data
