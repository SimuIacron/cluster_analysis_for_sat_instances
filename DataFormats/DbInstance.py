from util_scripts import DatabaseReader


class DbInstance:

    def __init__(self, features=None):

        if features is None:
            features = []
            features = features + DatabaseReader.FEATURES_BASE
            features = features + DatabaseReader.FEATURES_GATE
            features = features + DatabaseReader.FEATURES_SOLVER

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
        (self.un_sat, self.un_sat_wh), self.result_f = DatabaseReader.read_result_from_db()
        print("Queries finished")

        # -------------------------------------------------------------------
        # This section is a quickfix for the glucose_syrup solver which is wrongly with four times the real running time
        # section can be removed, after runtimes.db has been correctly updated

        idx = self.solver_f.index('glucose_syrup')
        for inst, inst_wh in zip(self.solver, self.solver_wh):
            if inst_wh[idx] != -1:
                inst[idx+1] = inst[idx+1] / 4
                inst_wh[idx] = inst_wh[idx] / 4

        # --------------------------------------------------------------------

        self.dataset_f, self.base_f, self.gate_f, self.solver_f, self.dataset, self.dataset_wh, self.base, self.base_wh,\
        self.gate, self.gate_wh, self.solver, self.solver_wh = self.generate_dataset(features)

        self.solver, removed_array = DatabaseReader.remove_empties(self.solver)

        self.solver_wh = DatabaseReader.remove_with_index_array(self.solver_wh, removed_array)
        self.gate = DatabaseReader.remove_with_index_array(self.gate, removed_array)
        self.gate_wh = DatabaseReader.remove_with_index_array(self.gate_wh, removed_array)
        self.base = DatabaseReader.remove_with_index_array(self.base, removed_array)
        self.base_wh = DatabaseReader.remove_with_index_array(self.base_wh, removed_array)
        self.family = DatabaseReader.remove_with_index_array(self.family, removed_array)
        self.family_wh = DatabaseReader.remove_with_index_array(self.family_wh, removed_array)
        self.un_sat = DatabaseReader.remove_with_index_array(self.un_sat, removed_array)
        self.un_sat_wh = DatabaseReader.remove_with_index_array(self.un_sat_wh, removed_array)

        print("remaining instances: " + str(len(self.solver)))

    # depending on which datasets where selected (currently base, gate and solver_time)
    # the dataset that is used for clustering is constructed and stored
    # in the self.dataset, self.dataset_wh and self.dataset_f variables
    # for later use
    def generate_dataset(self, features):

        dataset_f = features

        base_f = []
        gate_f = []
        solver_f = []

        for feature in features:
            if feature in DatabaseReader.FEATURES_BASE:
                base_f.append(feature)
            elif feature in DatabaseReader.FEATURES_GATE:
                gate_f.append(feature)
            elif feature in DatabaseReader.FEATURES_SOLVER:
                solver_f.append(feature)

        data = []
        data_wh = []
        base = []
        base_wh = []
        gate = []
        gate_wh = []
        solver = []
        solver_wh = []
        for idx, value in enumerate(self.base):
            data_wh.append([])
            data.append([self.base[idx][0]])
            base_wh.append([])
            base.append([self.base[idx][0]])
            gate_wh.append([])
            gate.append([self.base[idx][0]])
            solver_wh.append([])
            solver.append([self.base[idx][0]])

            for feature in features:
                if feature in DatabaseReader.FEATURES_BASE:
                    feature_idx = self.base_f.index(feature)
                    value = self.base_wh[idx][feature_idx]
                    data[idx].append(value)
                    data_wh[idx].append(value)
                    base[idx].append(value)
                    base_wh[idx].append(value)

                elif feature in DatabaseReader.FEATURES_GATE:
                    feature_idx = self.gate_f.index(feature)
                    value = self.gate_wh[idx][feature_idx]
                    data[idx].append(value)
                    data_wh[idx].append(value)
                    gate[idx].append(value)
                    gate_wh[idx].append(value)

                elif feature in DatabaseReader.FEATURES_SOLVER:
                    feature_idx = self.solver_f.index(feature)
                    value = self.solver_wh[idx][feature_idx]
                    data[idx].append(value)
                    data_wh[idx].append(value)
                    solver[idx].append(value)
                    solver_wh[idx].append(value)

        return dataset_f, base_f, gate_f, solver_f, data, data_wh, base, base_wh, gate, gate_wh, solver, solver_wh

    def get_complete_dataset(self):
        data = []
        data_wh = []
        dataset_f = []
        data.append(self.base)
        data_wh.append(self.base_wh)
        dataset_f = dataset_f + self.base_f

        data.append(self.gate)
        data_wh.append(self.gate_wh)
        dataset_f = dataset_f + self.gate_f

        # IMPORTANT: Solver data should always be the last data in the dataset,
        # otherwise some algorithms could break!
        data.append(self.solver)
        data_wh.append(self.solver_wh)
        dataset_f = dataset_f + self.solver_f

        dataset, dataset_wh = DatabaseReader.combine_queries(data, data_wh)
        return dataset, dataset_wh, dataset_f
