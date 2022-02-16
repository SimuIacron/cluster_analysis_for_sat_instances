from DataFormats import DatabaseReader
from DataFormats.DbInstance import DbInstance
from UtilScripts.scores import vbs, sbs, sbss, spar2, par2
from UtilScripts.util import get_combinations_of_databases
from UtilScripts.write_to_csv import write_to_csv
from run_experiments import read_json


def cleanup_solver_for_csv(solver_):
    return solver_.replace('_', '-')


version = '6'
input_file_clustering = 'clustering_general_v{ver}/standardscaler/general_clustering_{ver}_par2'.format(ver=version)
output_file = '/general_clustering_{ver}/'.format(ver=version)

data = read_json(input_file_clustering)

output, features = get_combinations_of_databases()
db_instance = DbInstance(features)

vbs_ = vbs(db_instance, DatabaseReader.TIMEOUT)
sbs_, sbs_name = sbs(db_instance, DatabaseReader.TIMEOUT)
sbss_, sbss_name = sbss(db_instance, DatabaseReader.TIMEOUT)

header = ['Score', 'Value (s)', 'Solver']
csv_data = [['SBS', sbs_, cleanup_solver_for_csv(sbs_name)],
            ['SBSS', sbss_, cleanup_solver_for_csv(sbss_name)],
            ['VBS', vbs_, '-']]

write_to_csv(output_file + 'general_scores', header, csv_data)

csv_solver = []
pseudo_cluster = list(range(len(db_instance.solver_wh)))
for solver in db_instance.solver_f:
    spar2_ = spar2(solver, db_instance, pseudo_cluster, DatabaseReader.TIMEOUT)
    par2_ = par2(solver, db_instance, pseudo_cluster, DatabaseReader.TIMEOUT)
    csv_solver.append([cleanup_solver_for_csv(solver), par2_, spar2_])

csv_solver = sorted(csv_solver, key=lambda d: d[1])
write_to_csv(output_file + 'solver_scores', ['Solver', 'Par2 (s)', 'SPar2 (s)'], csv_solver)
