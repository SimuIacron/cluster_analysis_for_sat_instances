import itertools
from run_plotting import plot_cbs_comparison
from run_plotting_histograms import plot_histograms_clustering, plot_boxplot_clustering

from util_scripts import DatabaseReader

sbs_file = 'vbs_sbs/sbs'
dpi = 192
angle = 20

temp_solver_features = DatabaseReader.FEATURES_SOLVER.copy()
temp_solver_features.pop(14)
temp_solver_features.pop(7)
input_dbs = [DatabaseReader.FEATURES_BASE, DatabaseReader.FEATURES_GATE, temp_solver_features]
output = [[DatabaseReader.FEATURES_BASE], [DatabaseReader.FEATURES_GATE],
          [DatabaseReader.FEATURES_BASE, DatabaseReader.FEATURES_GATE],
          [DatabaseReader.FEATURES_BASE, ['kissat']], [DatabaseReader.FEATURES_GATE, ['kissat']],
          [DatabaseReader.FEATURES_BASE, DatabaseReader.FEATURES_GATE, ['kissat']],
          [DatabaseReader.FEATURES_BASE, ['glucose']], [DatabaseReader.FEATURES_GATE, ['glucose']],
          [DatabaseReader.FEATURES_BASE, DatabaseReader.FEATURES_GATE, ['glucose']],
          [DatabaseReader.FEATURES_BASE, ['cadical']],
          [DatabaseReader.FEATURES_GATE, ['cadical']],
          [DatabaseReader.FEATURES_BASE, DatabaseReader.FEATURES_GATE, ['cadical']],
          input_dbs]
output_merged = []
for combination in output:
    comb = []
    for elem in combination:
        comb = comb + elem
    output_merged.append(comb)

# plot_histograms_clustering('single_solver/single_solver_par2',
#                            0, ['selected_data'],
#                            [output_merged],
#                            ['base', 'gate', 'base gate', 'base kissat', 'gate kissat', 'base gate kissat',
#                             'base glucose', 'gate glucose', 'base gate glucose',
#                             'base cadical', 'gate cadical',
#                             'base gate cadical', 'base gate runtimes'],
#                            max_cluster_amount=20, columns=3,
#                            bin_step=10, height=0.11, output_file='single_solver/hist_single_solver_20_new', normalize=True)
#
# plot_histograms_clustering('single_solver/single_solver_par2',
#                            0, ['selected_data'],
#                            [output_merged],
#                            ['base', 'gate', 'base gate', 'base kissat', 'gate kissat', 'base gate kissat',
#                             'base glucose', 'gate glucose', 'base gate glucose',
#                             'base cadical', 'gate cadical',
#                             'base gate cadical', 'base gate runtimes'],
#                            max_cluster_amount=100, columns=3,
#                            bin_step=10, height=0.11, output_file='single_solver/hist_single_solver_100_new', normalize=True)
#
# plot_histograms_clustering('single_solver/single_solver_all_algos_par2',
#                            0, ['selected_data'],
#                            [output_merged],
#                            ['base', 'gate', 'base gate', 'base kissat', 'gate kissat', 'base gate kissat',
#                             'base glucose', 'gate glucose', 'base gate glucose',
#                             'base cadical', 'gate cadical',
#                             'base gate cadical', 'base gate runtimes'],
#                            max_cluster_amount=20, columns=3,
#                            bin_step=10, height=0.11, output_file='single_solver/hist_single_solver_all_algos_20_new',
#                            normalize=True)
#

plot_histograms_clustering('single_solver/single_solver_all_algos_par2', sbs_file,
                           0, ['selected_data'],
                           [output_merged],
                           ['base', 'gate', 'base gate', 'base kissat', 'gate kissat', 'base gate kissat',
                            'base glucose', 'gate glucose', 'base gate glucose',
                            'base cadical', 'gate cadical',
                            'base gate cadical', 'base gate runtimes'],
                           max_cluster_amount=20, columns=3,
                           bin_step=10, height=0.11,
                           output_file='/single_runtime_feature_clustering/hist_single_runtime_features',
                           normalize=True, dpi=dpi)

output = [[DatabaseReader.FEATURES_BASE],
          [DatabaseReader.FEATURES_BASE, ['kissat']],
          [DatabaseReader.FEATURES_BASE, ['glucose']],
          [DatabaseReader.FEATURES_BASE, ['cadical']],

          [DatabaseReader.FEATURES_GATE],
          [DatabaseReader.FEATURES_GATE, ['kissat']],
          [DatabaseReader.FEATURES_GATE, ['glucose']],
          [DatabaseReader.FEATURES_GATE, ['cadical']],

          [DatabaseReader.FEATURES_BASE, DatabaseReader.FEATURES_GATE],
          [DatabaseReader.FEATURES_BASE, DatabaseReader.FEATURES_GATE, ['kissat']],
          [DatabaseReader.FEATURES_BASE, DatabaseReader.FEATURES_GATE, ['glucose']],
          [DatabaseReader.FEATURES_BASE, DatabaseReader.FEATURES_GATE, ['cadical']],

          input_dbs]

output_merged = []
for combination in output:
    comb = []
    for elem in combination:
        comb = comb + elem
    output_merged.append(comb)

plot_boxplot_clustering('single_solver/single_solver_all_algos_par2',
                        0, ['selected_data'],
                        [output_merged],
                        ['base', 'base kissat', 'base glucose', 'base cadical',
                         'gate', 'gate kissat', 'gate glucose', 'gate cadical',
                         'base gate', 'base gate kissat', 'base gate glucose', 'base gate cadical',
                         'base gate runtimes'],
                        max_cluster_amount=20,
                        output_file='/single_runtime_feature_clustering/box_single_runtime_features',
                        dpi=dpi, angle=angle)
