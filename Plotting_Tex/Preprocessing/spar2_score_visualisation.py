import os

from matplotlib import pyplot as plt

from PlottingAndEvaluationFunctions.func_stochastic_cluster_values import get_clustering_for_cluster
from DataFormats.DbInstance import DbInstance
from DataFormats import DatabaseReader
from util_scripts.scores import par2, spar2


def visualisation_spar2(data_clustering, cluster, db_instance: DbInstance, dpi=192, output_file='', show_plot=False):
    clustering = get_clustering_for_cluster(data_clustering, cluster)
    index_list = []
    for i, value in enumerate(clustering['clustering']):
        if value == cluster['cluster_idx']:
            index_list.append(i)

    csbs = cluster['cluster_par2'][0][0][0]
    csbss = cluster['cluster_performance_solver']

    csbs_par2 = par2(csbs, db_instance, index_list, DatabaseReader.TIMEOUT)
    csbs_spar2 = spar2(csbs, db_instance, index_list, DatabaseReader.TIMEOUT)

    csbss_par2 = par2(csbss, db_instance, index_list, DatabaseReader.TIMEOUT)
    csbss_spar2 = spar2(csbss, db_instance, index_list, DatabaseReader.TIMEOUT)

    plt.figure(figsize=(1200 / dpi, 1000 / dpi), dpi=dpi)
    
    X_axis = [csbs, csbss]

    plt.scatter(X_axis, [csbs_par2, csbss_par2], zorder=2, label='Par2')
    plt.scatter(X_axis, [csbs_spar2, csbss_spar2], zorder=2, label='SPar2')

    plt.ylabel('Solvers')
    plt.xlabel('(S)Par2-Score (s)')

    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.xticks(X_axis)

    if output_file != '':
        plt.savefig(os.environ['TEXPATH'] + output_file + '.svg')

    if show_plot:
        plt.show()




