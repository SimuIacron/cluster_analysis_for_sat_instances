import DatabaseReader


class DbInstance:

    def __init__(self):
        self.dataset = []
        self.dataset_wh = []

        print("Starting queries")
        self.base, self.base_wh = DatabaseReader.read_base_from_db()
        self.gate, self.gate_wh = DatabaseReader.read_gate_from_db()
        self.solver, self.solver_wh = DatabaseReader.read_solver_from_db()
        self.family, self.family_wh = DatabaseReader.read_family_from_db()
        print("Queries finished")

    def generate_dataset(self, selected_data):
        data = []
        data_wh = []
        if 'base' in selected_data:
            data.append(self.base)
            data_wh.append(self.base_wh)

        if 'gate' in selected_data:
            data.append(self.gate)
            data_wh.append(self.gate_wh)

        if 'solver' in selected_data:
            data.append(self.solver)
            data_wh.append(self.solver_wh)

        self.dataset, self.dataset_wh = DatabaseReader.combine_queries(data, data_wh)

