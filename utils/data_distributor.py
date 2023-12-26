import os
import json
import random
import numpy as np
import pandas as pd
from utils.helpers import read_product_label


class DataDistributor(object):
    def __init__(self, filename='distribution'):
        os.makedirs('utils/distribution/', exist_ok=True)
        self.filepath = f'utils/distribution/{filename}.json'
        self.distributions = {}

    def save_distributions(self, df_raw):
        df = df_raw[:]
        df = df.drop(columns=['ID'], errors='ignore')
        if '계약일자' in df.columns:
            df['계약일자'] = df['계약일자'].astype(int) / 10 ** 9

        for column in df.columns:
            column_data = df[column]

            if pd.api.types.is_object_dtype(column_data) or len(set(column_data)) < 20:
                self.distributions[column] = column_data.value_counts(normalize=True).to_dict()


            elif pd.api.types.is_numeric_dtype(column_data):
                self.distributions[column] = {'mean': column_data.mean(),
                                              'std': column_data.std()}
            else:
                print(f'{column}: Check data for distribution')

        with open(self.filepath, 'w') as file:
            json.dump(self.distributions, file)


class DataGenerator(object):
    def __init__(self, cust_dist_file, contract_dist_file, target_dist_file,
                 data_size=5000
                 ):
        self.cust_dist = self.load_distributions(f'utils/distribution/{cust_dist_file}.json')
        self.contract_dist = self.load_distributions(f'utils/distribution/{contract_dist_file}.json')
        self.target_dist = self.load_distributions(f'utils/distribution/{target_dist_file}.json')
        self.data_size = data_size

    def load_distributions(self, filepath):
        if os.path.exists(filepath):
            with open(filepath, 'rb') as file:
                distributions = json.load(file)
            return distributions
        else:
            print(f'File {filepath} not found. Generating new distributions')

    def generate_sample(self, distribution_dictionary, sample_size):
        sample = {}
        for column, dist_params in distribution_dictionary.items():
            if 'mean' in dist_params:
                sample[column] = np.random.normal(loc=dist_params['mean'], scale=dist_params['std'],
                                                  size=sample_size)
            else:
                values, probabilities = zip(*dist_params.items())
                sample[column] = np.random.choice(values, size=sample_size, p=probabilities)
        sample_df = pd.DataFrame(sample)
        return sample_df

    def make_virtual_data(self):
        cust_df = self.generate_sample(self.cust_dist, self.data_size)
        cust_df['마감년월'] = cust_df['마감년월'].astype(int)
        cust_df['ID'] = list(range(1, self.data_size + 1))
        assert len(set(cust_df['마감년월'])) == 1
        assert cust_df['ID'].is_unique

        reference_date = pd.to_datetime(cust_df['마감년월'][0], format='%Y%m')

        contract_df = self.generate_sample(self.contract_dist, self.data_size * 10)
        contract_df['ID'] = list(range(1, self.data_size + 1)) * 10
        contract_df['계약일자'] = pd.to_datetime(contract_df['계약일자'] // 10 ** 9, unit='s')
        mask = contract_df['계약일자'] > reference_date
        contract_df.loc[mask, '계약일자'] = np.random.choice(pd.date_range(end=reference_date, periods=sum(mask)),
                                                         size=sum(mask))
        contract_df['prdt_cat'] = contract_df['상품중분류2'].replace(read_product_label()['contract_previous_label'])

        target_df = self.generate_sample(self.target_dist, self.data_size * 5)
        target_df['ID'] = list(range(1, self.data_size + 1)) * 5
        target_df['계약일자'] = pd.to_datetime(target_df['계약일자'] // 10 ** 9, unit='s')
        mask = target_df['계약일자'] < reference_date
        target_df.loc[mask, '계약일자'] = np.random.choice(
            pd.date_range(start=reference_date, end=reference_date + pd.DateOffset(months=6)),
            size=sum(mask))
        target_df['prdt_cat'] = target_df['상품중분류2'].replace(read_product_label()['contract_target_label'])
        target_df = target_df.drop_duplicates(subset=['prdt_cat', 'ID'])
        target_labels = set(read_product_label()['contract_target_label'].values())
        assert target_labels == set(target_df.prdt_cat), "missing value in 'prdt_cat'"

        assert set(cust_df.ID) == set(contract_df.ID) == set(target_df.ID)
        return cust_df, contract_df, target_df


if __name__ == '__main__':
    # 1. DataDistributor
    data = {
        'column1': np.random.normal(0, 1, 100),
        'column2': np.random.uniform(0, 1, 100),
        'column3': np.random.choice(['A', 'B', 'C'], 100),
    }
    df = pd.DataFrame(data)
    distributor = DataDistributor()
    distributor.save_distributions(df)

    # 2. DataGenerator
    dg = DataGenerator('dev_customer_dist', 'dev_contract_dist', 'dev_target_dist')
    cust_df, contract_df, target_df = dg.make_virtual_data()
    print(cust_df.head())
