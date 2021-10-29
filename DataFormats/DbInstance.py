import DatabaseReader


class DbInstance:

    def __init__(self):
        print("Starting queries")
        self.instances, self.instances_wh = DatabaseReader.read_from_db(DatabaseReader.FEATURES_BASE +
                                                                        DatabaseReader.FEATURES_GATE)
        self.solver, self.solver_wh = DatabaseReader.read_solver_from_db()
        self.family, self.family_wh = DatabaseReader.read_family_from_db()
        print("Queries finished")
