from DataFormats.DbInstance import DbInstance
from DataFormats import DatabaseReader
from UtilScripts.scores import par2, spar2
from UtilScripts.util import get_combinations_of_databases

output_merged, features = get_combinations_of_databases()

db_instance = DbInstance(features)

version = '6'
output_file = '/general_clustering_{ver}/'.format(ver=version)

solver_list = []
for solver in db_instance.solver_f:
    par2_ = par2(solver, db_instance, range(0, len(db_instance.solver_wh)), DatabaseReader.TIMEOUT)
    spar2_ = spar2(solver, db_instance, range(0, len(db_instance.solver_wh)), DatabaseReader.TIMEOUT)
    solver_list.append([solver, par2_, spar2_])


sort = sorted(solver_list, key=lambda d: d[1])

pass

