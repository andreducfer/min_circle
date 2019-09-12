import numpy as np


class Instance:
    def __init__(self, instance_path, solution_path, dataset_name):
        self.instance_path = instance_path
        self.solution_path = solution_path
        self.dataset_name = dataset_name

        self._load_dataset(self.instance_path)

    def _load_dataset(self, instance_path):
        # Find number of lines in dataset
        self.number_points = len(open(instance_path, encoding='utf-8').readlines(  ))

        # Create numpy array with the numer of elements of my dataset
        self.data_values = np.zeros((self.number_points, 2))

        # Populate my array with elements in dataset
        with open(instance_path, encoding='utf-8') as fp:
            for i, line in enumerate(fp):
                values = line.split()
                self.data_values[i][0] = np.float(values[0])
                self.data_values[i][1] = np.float(values[1])
