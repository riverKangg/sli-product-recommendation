import warnings
warnings.filterwarnings('ignore')

from utils import *
from data_distributor import DataDistributor
from data_builder import DataBuilder

seed = 42
random.seed(seed)

class TargetProcessor(object):
    def __init__(self,
                 customer_dataset,
                 contract_previous_dataset,
                 contract_target_dataset,
                 target_label='target',
                 development_reference_date='202305',
                 ):
        print(f'■■■ Target Processing ■■■')
        self.customer_dataset = customer_dataset

        assert len(set(customer_dataset.마감년월)) == 1
        self.yyyymm = list(set(customer_dataset.마감년월))[0]
        print(f' reference date: {self.yyyymm}')
        self.is_development = True if str(self.yyyymm) == development_reference_date else False
        print(f' is_development: {self.is_development}')

        self.contract_target_dataset = contract_target_dataset
        self.contract_target_y = contract_target_dataset[['ID', 'prdt_cat']].drop_duplicates()

        self.product_label = read_product_label()['contract_target_label']
        self.target_label = target_label

        assert set(self.product_label.values()) == set(self.contract_target_y.prdt_cat)

    def sampling_non_target_data(self):
        if self.is_development:
            print(' Start sampling non-target data')
            tot_ids = set(self.contract_target_y.ID)
            contract_target_n = pd.DataFrame()
            print('  Code: target / non-target')
            for name, code in self.product_label.items():
                code_target_data = self.contract_target_y[self.contract_target_y['prdt_cat'] == code].drop_duplicates()
                code_target_ids = set(code_target_data.ID)
                assert len(code_target_ids) == len(set(code_target_ids))

                categroy_non_target_data = self.contract_target_y.loc[
                    ~self.contract_target_y['ID'].isin(code_target_ids), ['ID']].drop_duplicates()

                num_of_target = len(code_target_ids)
                num_of_non_target = len(tot_ids - code_target_ids)

                print(f'  - {code} {name}: {num_of_target:,} / {num_of_non_target:,}')
                num_of_sample = min(num_of_non_target, num_of_target * 2)
                categroy_non_target_data = categroy_non_target_data.sample(num_of_sample, replace=False,
                                                                           random_state=seed)
                categroy_non_target_data['prdt_cat'] = code

                contract_target_n = contract_target_n.append(categroy_non_target_data)

                del categroy_non_target_data

            print(f'tot_num: {contract_target_n.shape[0]:,}')
            assert len(contract_target_n[contract_target_n.duplicated()]) == 0
            self.contract_target_n = contract_target_n

    def make_target_data(self):
        self.sampling_non_target_data()

        self.contract_target_y[self.target_label] = 1
        if self.is_development:
            self.contract_target_n[self.target_label] = 0
            data_target_label = self.contract_target_y.append(self.contract_target_n)
        else:
            data_target_label = self.contract_target_y[:]

        data_target_label = data_target_label.set_index('ID')
        data_target_label.columns = ['target_category', self.target_label]
        self.data_target_label = data_target_label
        return data_target_label

    def make_target_describe(self):
        # count
        traget_cnt_dict = self.contract_target_y['prdt_cat'].value_counts().to_dict()

        # percent
        total_count = sum(traget_cnt_dict.values())
        target_precent = {}
        for k, v in traget_cnt_dict.items():
            target_precent[k] = round(v / total_count * 100, 2)
        target_precent = {key: value for key, value in sorted(target_precent.items(), key=lambda x: x[1], reverse=True)}

        return traget_cnt_dict, target_precent

    @property
    def return_target_describe(self):
        num_of_target = self.contract_target_y.shape[0]
        num_of_rows = self.data_target_label.shape[0]
        num_of_target_ids = len(set(self.contract_target_y.ID))
        num_of_ids = len(set(self.contract_target_y.ID))
        traget_cnt_dict, target_precent = self.make_target_describe()

        target_describe = {'target_label': self.target_label,
                           'target_product_categroy': 'target_category',
                           'num_of_target': num_of_target,
                           'percent_of_target': round(num_of_target / num_of_rows * 100, 1),
                           'num_of_target_ids': num_of_target_ids,
                           'percent_of_target_ids': round(num_of_target_ids / num_of_ids * 100, 1),
                           'target_category_count': traget_cnt_dict,
                           'target_category_percent': target_precent,
                           }
        return target_describe


if __name__ == '__main__':
    customer_distributor = DataDistributor('dev_customer_dist')
    customer_df = customer_distributor.generate_samples()


    data_builder()
    tp = TargetProcessor(data_dict['dev']['dev_customer'], data_dict['dev']['dev_contract_previous'],
                         data_dict['dev']['dev_contract_target'])
    target_data = tp.make_target_data()