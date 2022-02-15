import os
from matplotlib import pyplot as plt
from DataFormats.DbInstance import DbInstance


# For a clustering plots the size of each cluster together with the share of the biggest family and some infos for the
# biggest family in each cluster
def plot_family_distribution_of_clusters(data_clusters, db_instance: DbInstance, output_file='',
                                         show_plot=False, dpi=192):
    biggest_family_size = []
    biggest_family_name = []
    other_families_size = []
    percentage_familes = []

    for cluster in data_clusters:

        biggest_family_name.append(cluster['family_list'][0][0])
        biggest_family_size.append(cluster['family_list'][0][1])
        percentage_familes.append(cluster['family_total_percentage'])
        size = 0
        for family in cluster['family_list'][1:]:
            size = size + family[1]
        other_families_size.append(size)

    X_axis = range(len(biggest_family_size))

    plt.figure(figsize=(1200 / dpi, 1000 / dpi), dpi=dpi)
    plt.barh(X_axis, biggest_family_size, label='Biggest Family')
    plt.barh(X_axis, other_families_size, label='Other Families', left=biggest_family_size)

    for x, family in zip(X_axis, biggest_family_name):
        plt.annotate('{family} ({size}/{total_size}), {percentage}% of family'.format(family=family, size=biggest_family_size[x],
                                                             total_size=biggest_family_size[x] + other_families_size[
                                                                 x], percentage=round(percentage_familes[x] * 100)),
                     (150, x+0.1), rotation=0)

    plt.legend(loc='upper right')
    plt.ylabel('Cluster')
    plt.xlabel('Size')
    plt.tight_layout()
    plt.yticks(X_axis)
    plt.gca().invert_yaxis()

    if output_file != '':
        plt.savefig(os.environ['TEXPATH'] + output_file + '.svg')

    if show_plot:
        plt.show()


# for a clustering plots the csbs score of each cluster and the speed of the sbs on the cluster.
# annotates the csbs as well as the solvers in the strip of the csbs
def plot_runtime_comparison_sbs(data_cluster, sbs_solver, output_file='', show_plot=False, dpi=192):
    runtimes_sbs = []
    runtimes_cluster_solver = []
    solvers = [cluster['par2_strip'] for cluster in data_cluster]
    for cluster in data_cluster:

        # cluster_solver = cluster['cluster_par2']
        runtimes_cluster_solver.append(cluster['cluster_par2'][0][0][1])

        for solver_tuple in cluster['cluster_par2'][0]:
            if solver_tuple[0] == sbs_solver:
                runtimes_sbs.append(solver_tuple[1])

    X_axis = range(len(runtimes_sbs))

    plt.figure(figsize=(1200 / dpi, 1000 / dpi), dpi=dpi)

    plt.scatter(runtimes_sbs, X_axis, zorder=2, label='SBS')
    plt.barh(X_axis, runtimes_cluster_solver, zorder=1, label='CSBS')

    plt.ylabel('Cluster')
    plt.xlabel('Par2-Score (s)')

    for x, strip in zip(X_axis, solvers):
        plt.annotate(strip[0][0], (1000, x+0.1), rotation=0, weight='bold')
        strip_text = ''
        for i, current_solver in enumerate(strip[1:]):
            if i < 5:
                strip_text = strip_text + ', ' + current_solver[0]
            else:
                strip_text = strip_text + ', ' + str(len(strip[1:]) - i) + ' more solvers'
                break

        plt.annotate(strip_text, (1000 + 92 * len(strip[0][0]), x+0.1), rotation=0)

    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.yticks(X_axis)
    plt.gca().invert_yaxis()

    if output_file != '':
        plt.savefig(os.environ['TEXPATH'] + output_file + '.svg')

    if show_plot:
        plt.show()

