from sklearn.metrics import normalized_mutual_info_score
from PlottingAndEvaluationFunctions.func_plot_changes_between_clusterings import filter_clustering_settings, \
    filter_clustering_cluster_number
from DataFormats.DbInstance import DbInstance
from run_experiments import read_json
from DataFormats import DatabaseReader
from UtilScripts.scores import clustering_best, spar2
from UtilScripts.util import get_combinations_of_databases

version = '6'
input_file_clustering = 'clustering_general_v{ver}/standardscaler/general_clustering_{ver}_par2'.format(ver=version)
print(input_file_clustering)

output_merged, features = get_combinations_of_databases()

db_instance = DbInstance(features)

data = read_json(input_file_clustering)
filtered = filter_clustering_settings(data, ['selected_data', 'cluster_algorithm'],
                           [
                               [DatabaseReader.FEATURES_BASE, DatabaseReader.FEATURES_GATE,
                                DatabaseReader.FEATURES_BASE + DatabaseReader.FEATURES_GATE],
                               ['DBSCAN']
                           ])
filtered2 = filter_clustering_cluster_number(filtered, 40)


cbss_clusterings = []

for clustering in filtered:
    cbss = clustering_best(clustering['clustering'], db_instance, spar2, DatabaseReader.TIMEOUT)
    family = [item[0] for item in db_instance.family_wh]
    nmi = normalized_mutual_info_score(clustering['clustering'], family)
    new_dict = dict(clustering, **{
        'cbss': cbss,
        'nmi': nmi
    })
    cbss_clusterings.append(new_dict)


sort1 = sorted(cbss_clusterings, key=lambda d: d['par2'][0])
sort2 = sorted(cbss_clusterings, key=lambda d: d['cbss'])
sort3 = sorted(cbss_clusterings, key=lambda d: d['nmi'], reverse=True)

pass
