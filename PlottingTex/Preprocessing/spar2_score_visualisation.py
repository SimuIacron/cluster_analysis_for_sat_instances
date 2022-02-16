import os

from matplotlib import pyplot as plt

from PlottingAndEvaluationFunctions.func_stochastic_cluster_values import get_clustering_for_cluster
from DataFormats.DbInstance import DbInstance
from DataFormats import DatabaseReader
from UtilScripts.scores import par2, spar2


def visualisation_spar2(data_clustering, cluster, db_instance: DbInstance, dpi=192, output_file='', show_plot=False):
    clustering = get_clustering_for_cluster(data_clustering, cluster)
    csbs = cluster['cluster_par2'][0][0][0]
    csbss = cluster['cluster_performance_solver']
    csbs_index = db_instance.solver_f.index(csbs)
    csbss_index = db_instance.solver_f.index(csbss)

    index_list = []
    csbs_runtimes = []
    csbss_runtimes = []
    for i, value in enumerate(clustering['clustering']):
        if value == cluster['cluster_idx']:
            index_list.append(i)
            csbs_runtimes.append(db_instance.solver_wh[i][csbs_index])
            csbss_runtimes.append(db_instance.solver_wh[i][csbss_index])

    runtime_data = [csbs_runtimes, csbss_runtimes]

    csbs_par2 = par2(csbs, db_instance, index_list, DatabaseReader.TIMEOUT)
    csbs_spar2 = spar2(csbs, db_instance, index_list, DatabaseReader.TIMEOUT)

    csbss_par2 = par2(csbss, db_instance, index_list, DatabaseReader.TIMEOUT)
    csbss_spar2 = spar2(csbss, db_instance, index_list, DatabaseReader.TIMEOUT)

    plt.figure(figsize=(500 / dpi, 700 / dpi), dpi=dpi)
    
    X_axis = [csbs + ' (CSBS)', csbss + ' (CSBSS)']

    plt.boxplot(runtime_data, zorder=1)
    plt.scatter([1, 2], [csbs_par2, csbss_par2], zorder=2, label='Par2')
    plt.scatter([1, 2], [csbs_spar2, csbss_spar2], zorder=2, label='SPar2')


    plt.xlabel('Solvers')
    plt.ylabel('(S)Par2-Score (s)')

    plt.legend(loc='upper center')
    plt.tight_layout()
    plt.xticks(range(1, len(X_axis) + 1), X_axis)

    if output_file != '':
        plt.savefig(os.environ['TEXPATH'] + output_file + '.svg')

    if show_plot:
        plt.show()




