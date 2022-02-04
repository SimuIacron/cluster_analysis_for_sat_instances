from ClusteringAnalysis.plot_changes_between_clusterings import filter_clustering_settings
from run_experiments import read_json
from util_scripts import DatabaseReader

version = '6'
input_file_clustering = 'clustering_general_v{ver}/standardscaler/general_clustering_{ver}_par2'.format(ver=version)
print(input_file_clustering)

data = read_json(input_file_clustering)
filtered = filter_clustering_settings(data, ['selected_data', 'cluster_algorithm'],
                           [
                               [DatabaseReader.FEATURES_BASE, DatabaseReader.FEATURES_GATE,
                                DatabaseReader.FEATURES_BASE + DatabaseReader.FEATURES_GATE],
                               ['KMEANS']
                           ])

sort = sorted(filtered, key=lambda d: d['par2'][0])

pass
