import DatabaseReader


class DbInstance:

    def __init__(self):
        # stores the values of the constructed set of features
        self.dataset = []
        # stores the values without the hash of the constructed set of features
        self.dataset_wh = []
        # saves the feature names of the constructed set of features
        self.dataset_f = []

        print("Starting queries")
        # get all basic values and features
        # tuple needed, because of format of the output
        (self.base, self.base_wh), self.base_f = DatabaseReader.read_base_from_db()
        (self.gate, self.gate_wh), self.gate_f = DatabaseReader.read_gate_from_db()
        (self.solver, self.solver_wh), self.solver_f, = DatabaseReader.read_solver_from_db()
        (self.family, self.family_wh), self.family_f = DatabaseReader.read_family_from_db()
        print("Queries finished")

    # depending on which datasets where selected (currently base, gate and solver_time)
    # the dataset that is used for clustering is constructed and stored
    # in the self.dataset, self.dataset_wh and self.dataset_f variables
    # for later use
    def generate_dataset(self, selected_data):
        data = []
        data_wh = []
        self.dataset_f = []
        if 'base' in selected_data:
            data.append(self.base)
            data_wh.append(self.base_wh)
            self.dataset_f = self.dataset_f + self.base_f

        if 'gate' in selected_data:
            data.append(self.gate)
            data_wh.append(self.gate_wh)
            self.dataset_f = self.dataset_f + self.gate_f

        # IMPORTANT: Solver data should always be the last data in the dataset,
        # otherwise some algorithms could break!
        if 'solver' in selected_data:
            data.append(self.solver)
            data_wh.append(self.solver_wh)
            self.dataset_f = self.dataset_f + self.solver_f

        self.dataset, self.dataset_wh = DatabaseReader.combine_queries(data, data_wh)

