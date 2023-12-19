import os
import json
import numpy as np
import pandas as pd

class DataDistributor(object):
    def __init__(self, filename='distribution'):
        os.makedirs('utils/distribution/', exist_ok=True)
        self.filepath = f'utils/distribution/{filename}.json'
        self.distributions = {}

    def save_distributions(self, df):
        for column in df.columns:
            column_data = df[column]
            if pd.api.types.is_numeric_dtype(column_data):
                self.distributions[column] = {'mean': column_data.mean(),
                                              'std': column_data.std()}
            else:
                self.distributions[column] = column_data.value_counts(normalize=True).to_dict()

        with open(self.filepath, 'w') as file:
            json.dump(self.distributions, file)

    def load_distributions(self):
        if os.path.exists(self.filepath):
            with open(self.filepath, 'rb') as file:
                self.distributions = json.load(file)
        else:
            print(f'File {self.filepath} not found. Generating new distributions')

    def generate_samples(self, sample_size=100):
        if not self.distributions:
            self.load_distributions()

        generated_data = {}
        for column, dist_params in self.distributions.items():
            if 'mean' in dist_params:
                generated_data[column] = np.random.normal(loc=dist_params['mean'], scale=dist_params['std'],
                                                          size=sample_size)
            else:
                values, probabilities = zip(*dist_params.items())
                generated_data[column] = np.random.choice(values, size=sample_size, p=probabilities)

        generated_df = pd.DataFrame(generated_data)
        generated_df['ID'] = range(1, sample_size + 1)
        return generated_df


if __name__ == '__main__':
    data = {
        'column1': np.random.normal(0, 1, 100),
        'column2': np.random.uniform(0, 1, 100),
        'column3': np.random.choice(['A', 'B', 'C'], 100),
    }
    df = pd.DataFrame(data)
    distributor = DataDistributor()
    # distributor.save_distributions(df)

    dev_distributor = DataDistributor('dev_customer_dist')
    dev_sample_df = dev_distributor.generate_samples()

    print('Development Set Generating:')
    print(dev_sample_df.head())