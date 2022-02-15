from numpy import mean
from sklearn.metrics import normalized_mutual_info_score

from PlottingAndEvaluationFunctions.func_plot_changes_between_clusterings import calculate_homogeneity_completeness_of_clustering_to_family
from DataFormats.DbInstance import DbInstance
from run_experiments import read_json
from DataFormats import DatabaseReader
from UtilScripts.scores import cbs, cbss, sbs, sbss, vbs, vbss
from UtilScripts.util import get_combinations_of_databases

version = '6'
input_file_clustering = 'clustering_general_v{ver}/standardscaler/general_clustering_{ver}_par2'.format(ver=version)
print(input_file_clustering)
id_ = 33

data = read_json(input_file_clustering)

clustering = data[id_]
assert clustering['id'] == id_, 'expected id to match the index'

output, features = get_combinations_of_databases()
db_instance = DbInstance(features)

homogeneity, completeness = calculate_homogeneity_completeness_of_clustering_to_family(clustering['clustering'], db_instance)
homogeneity_list = [item for _, item in homogeneity.items()]
completeness_list = [item for _, item in completeness.items()]
homogeneity_mean = mean(homogeneity_list)
completeness_mean = mean(completeness_list)

print('homogeneity mean: {a}'.format(a=homogeneity_mean))
print('completeness mean: {a}'.format(a=completeness_mean))


family = [item[0] for item in db_instance.family_wh]

nmi = normalized_mutual_info_score(clustering['clustering'], family)
cbs_ = cbs(clustering['clustering'], db_instance, DatabaseReader.TIMEOUT)
cbss_ = cbss(clustering['clustering'], db_instance, DatabaseReader.TIMEOUT)
sbs_, sbs_name = sbs(db_instance, DatabaseReader.TIMEOUT)
sbss_, sbss_name = sbss(db_instance, DatabaseReader.TIMEOUT)
vbs_ = vbs(db_instance, DatabaseReader.TIMEOUT)
vbss_ = vbss(db_instance, DatabaseReader.TIMEOUT)

print('NMI: {a}'.format(a=nmi))
print('CBS: {a}'.format(a=cbs_))
print('CBSS: {a}'.format(a=cbss_))
print('SBS: {a}, {b}'.format(a=sbs_, b=sbs_name))
print('SBSS: {a}, {b}'.format(a=sbss_, b=sbss_name))
print('VBS: {a}'.format(a=vbs_))
print('VBSS: {a}'.format(a=vbss_))



