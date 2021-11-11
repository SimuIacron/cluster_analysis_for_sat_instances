import exportFigure
from DataAnalysis import scoring
from DataFormats.DbInstance import DbInstance
from sklearn.metrics import normalized_mutual_info_score, adjusted_mutual_info_score

print(scoring.create_contingency_matrix([0,1,2,2,1,1], [0,0,1,0,2,3]))

db_instance = DbInstance()

family_int = scoring.convert_families_to_int(db_instance.family_wh)
solver_int = scoring.get_best_solver_int(db_instance)

print(adjusted_mutual_info_score(family_int, solver_int))

