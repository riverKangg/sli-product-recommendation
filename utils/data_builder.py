import json
import random
import pandas as pd
from dateutil.relativedelta import relativedelta

from utils.helpers import read_product_label


class DataBuilder(object):
    def __init__(self,
                 dev_date='202305',
                 oot_date='202308',
                 target_label='target',
                 product_category='상품중분류2',
                 data_name='base',
                 drop_category=None,
                 **kwargs):
        self.dev_date = dev_date
        self.oot_date = oot_date
        self.target_label = target_label
        self.product_category = product_category
        self.data_name = data_name

        if 'small_category' in data_name:
            self.drop_category = ['42.변액연금', '12.변액종신', '52.변액저축', '20.CI', '35.정기', '33.상해']
        elif 'del_variable' in data_name:
            self.drop_category = ['42.변액연금', '12.변액종신', '52.변액저축']
        else:
            self.drop_category = drop_category

        self.seed = 42
        random.seed(self.seed)

    def load_dataset(self, yyyymm):
        print('■■■ Load Data ■■■')
        print(f' reference date : {yyyymm}')
        customer = pd.read_csv(f'data/CUST_BASE_{yyyymm}.csv', encoding='euc-kr', converters={'ID': str})
        check_customer = list(filter(lambda x: x not in customer.columns, ['ID']))
        if check_customer:
            raise ValueError(f'The customer dataset must include an {check_customer} column.')

        contract = pd.read_csv(f'data/PROD_BASE_{yyyymm}.csv', encoding='euc-kr', converters={'ID': str}).dropna()
        check_contract = list(
            filter(lambda x: x not in contract.columns, ['ID', '계약일자', self.target_label, self.product_category]))
        if check_customer:
            raise ValueError(f'The contract dataset must include an {check_contract} column.')
        contract['계약일자'] = pd.to_datetime(contract['계약일자'], format='%Y%m%d')

        return customer, contract

    def contract_split(self, yyyymm, contract):
        split_date = pd.to_datetime(yyyymm, format='%Y%m') + relativedelta(months=1)
        print(f' split date: {split_date}')
        # for train
        contract_previous = contract[pd.to_datetime(contract['계약일자'], format='%Y%m%d') < split_date]
        # for target
        contract_target = contract[pd.to_datetime(contract['계약일자'], format='%Y%m%d') >= split_date]
        len_contract_target_before = contract_target.shape[0]
        contract_target = contract_target[~contract_target.상품분류.isin(['34.실손', '80.미니보험'])]
        contract_target = contract_target[~contract_target.상품중분류2.isin(['34.실손', '36.단체', '기타'])]
        if self.drop_category is not None:
            contract_target = contract_target[~contract_target.상품중분류2.isin(self.drop_category)]
        contract_target = contract_target.drop_duplicates()

        # check split dataset
        assert max(contract_previous.계약일자) < min(contract_target.계약일자)
        target_duration = (max(contract_target.계약일자) - min(contract_target.계약일자)) / pd.Timedelta(days=1) // 30
        if yyyymm == self.dev_date:
            assert target_duration == 5

        # label encoding
        if self.dev_date == yyyymm:
            self.save_product_label(contract_previous, contract_target)
        contract_previous, contract_target = self.contract_label_encoding(contract_previous, contract_target)

        # keep only necessary columns
        necessary_columns = ['ID', self.product_category, '계약일자', 'prdt_cat']
        contract_previous = contract_previous[necessary_columns].drop_duplicates()

        contract_target = contract_target[necessary_columns].drop_duplicates()

        # print result
        print('■■■ Contract Split ■■■')
        print(f' Number of total contracts: {contract.shape[0]:,}')
        print(f' Number of previous contracts: {contract_previous.shape[0]:,}')
        print(f' Number of target contract: {len_contract_target_before:,} → {contract_target.shape[0]:,}')
        print('  Check the duration')
        print(f'  - previous: {contract_previous.계약일자.min()} ~ {contract_previous.계약일자.max()}')
        print(f'  - target: {contract_target.계약일자.min()} ~ {contract_target.계약일자.max()}')

        return contract_previous, contract_target

    def save_product_label(self, contract_previous, contract_target):
        contract_target_names = list(contract_target[self.product_category].value_counts().keys())
        contract_target_codes = list(range(1, len(contract_target_names) + 1))  # 0 is considered as null
        contract_target_label = dict(zip(contract_target_names, contract_target_codes))

        contract_previous_names = pd.Series(contract_previous[self.product_category].value_counts().keys())
        added_contract_previous_names = contract_previous_names[
            ~pd.Series(contract_previous_names).isin(contract_target_names)]
        start_added = max(contract_target_codes)
        added_contract_previous_codes = list(
            range(start_added + 1, start_added + len(added_contract_previous_names) + 1))
        added_contract_previous_label = dict(zip(added_contract_previous_names, added_contract_previous_codes))

        contract_previous_label = {**contract_target_label, **added_contract_previous_label}

        label_dictionary = {'contract_previous_label': contract_previous_label,
                            'contract_target_label': contract_target_label}

        with open('utils/target_label_dictionary.json', 'w') as f:
            json.dump(label_dictionary, f)

    def contract_label_encoding(self, contract_previous, contract_target):
        label_dictionary = read_product_label()
        contract_previous_label = label_dictionary['contract_previous_label']
        contract_target_label = label_dictionary['contract_target_label']

        contract_previous['prdt_cat'] = contract_previous[self.product_category].replace(contract_previous_label)
        contract_target['prdt_cat'] = contract_target[self.product_category].replace(contract_target_label)
        return contract_previous, contract_target

    def revise_ids(self, customer, contract_previous, contract_target):
        target_true_ids = set(contract_target.ID)
        previous_true_ids = set(contract_previous.ID)
        customer_ids = set(customer.ID)
        common_ids = list(target_true_ids.intersection(customer_ids).intersection(previous_true_ids))

        customer_new = customer[customer.ID.isin(common_ids)]
        contract_previous_new = contract_previous[contract_previous.ID.isin(common_ids)]
        contract_target_new = contract_target[contract_target.ID.isin(common_ids)]

        print('■■■ Find common IDs ■■■')
        print(f' Customer set: {len(customer):,} → {len(customer_new):,}')
        print(f' Contract Previous set: {len(contract_previous):,} → {len(contract_previous_new):,}')
        print(f' Contract Target set: {len(contract_target):,} → {len(contract_target_new):,}\n')

        return customer_new, contract_previous_new, contract_target_new

    def build_dataset(self):
        dev_customer_raw, dev_contract_raw = self.load_dataset(self.dev_date)
        dev_contract_previous, dev_contract_target = self.contract_split(self.dev_date, dev_contract_raw)
        dev_customer, dev_contract_previous, dev_contract_target = self.revise_ids(dev_customer_raw,
                                                                                   dev_contract_previous,
                                                                                   dev_contract_target)

        oot_customer_raw, oot_contract_raw = self.load_dataset(self.oot_date)
        oot_contract_previous, oot_contract_target = self.contract_split(self.oot_date, oot_contract_raw)
        oot_customer, oot_contract_previous, oot_contract_target = self.revise_ids(oot_customer_raw,
                                                                                   oot_contract_previous,
                                                                                   oot_contract_target)

        return {'dev': {'customer': dev_customer,
                        'contract': dev_contract_previous,
                        'target': dev_contract_target},
                'oot': {'customer': oot_customer,
                        'contract': oot_contract_previous,
                        'target': oot_contract_target}}


if __name__ == '__main__':
    data_builder = DataBuilder(data_name='del_variable')
    data_builder.build_dataset()

import json
import random
import pandas as pd
from dateutil.relativedelta import relativedelta

from utils.helpers import read_product_label


class DataBuilder(object):
    def __init__(self,
                 dev_date='202305',
                 oot_date='202308',
                 target_label='target',
                 product_category='상품중분류2',
                 data_name='base',
                 drop_category=None,
                 **kwargs):
        self.dev_date = dev_date
        self.oot_date = oot_date
        self.target_label = target_label
        self.product_category = product_category
        self.data_name = data_name

        if 'small_category' in data_name:
            self.drop_category = ['42.변액연금', '12.변액종신', '52.변액저축', '20.CI', '35.정기', '33.상해']
        elif 'del_variable' in data_name:
            self.drop_category = ['42.변액연금', '12.변액종신', '52.변액저축']
        else:
            self.drop_category = drop_category

        self.seed = 42
        random.seed(self.seed)

    def load_dataset(self, yyyymm):
        print('■■■ Load Data ■■■')
        print(f' reference date : {yyyymm}')
        customer = pd.read_csv(f'data/CUST_BASE_{yyyymm}.csv', encoding='euc-kr', converters={'ID': str})
        check_customer = list(filter(lambda x: x not in customer.columns, ['ID']))
        if check_customer:
            raise ValueError(f'The customer dataset must include an {check_customer} column.')

        contract = pd.read_csv(f'data/PROD_BASE_{yyyymm}.csv', encoding='euc-kr', converters={'ID': str}).dropna()
        check_contract = list(
            filter(lambda x: x not in contract.columns, ['ID', '계약일자', self.target_label, self.product_category]))
        if check_customer:
            raise ValueError(f'The contract dataset must include an {check_contract} column.')
        contract['계약일자'] = pd.to_datetime(contract['계약일자'], format='%Y%m%d')

        return customer, contract

    def contract_split(self, yyyymm, contract):
        split_date = pd.to_datetime(yyyymm, format='%Y%m') + relativedelta(months=1)
        print(f' split date: {split_date}')
        # for train
        contract_previous = contract[pd.to_datetime(contract['계약일자'], format='%Y%m%d') < split_date]
        # for target
        contract_target = contract[pd.to_datetime(contract['계약일자'], format='%Y%m%d') >= split_date]
        len_contract_target_before = contract_target.shape[0]
        contract_target = contract_target[~contract_target.상품분류.isin(['34.실손', '80.미니보험'])]
        contract_target = contract_target[~contract_target.상품중분류2.isin(['34.실손', '36.단체', '기타'])]
        if self.drop_category is not None:
            contract_target = contract_target[~contract_target.상품중분류2.isin(self.drop_category)]
        contract_target = contract_target.drop_duplicates()

        # check split dataset
        assert max(contract_previous.계약일자) < min(contract_target.계약일자)
        target_duration = (max(contract_target.계약일자) - min(contract_target.계약일자)) / pd.Timedelta(days=1) // 30
        if yyyymm == self.dev_date:
            assert target_duration == 5

        # label encoding
        if self.dev_date == yyyymm:
            self.save_product_label(contract_previous, contract_target)
        contract_previous, contract_target = self.contract_label_encoding(contract_previous, contract_target)

        # keep only necessary columns
        necessary_columns = ['ID', self.product_category, '계약일자', 'prdt_cat']
        contract_previous = contract_previous[necessary_columns].drop_duplicates()

        contract_target = contract_target[necessary_columns].drop_duplicates()

        # print result
        print('■■■ Contract Split ■■■')
        print(f' Number of total contracts: {contract.shape[0]:,}')
        print(f' Number of previous contracts: {contract_previous.shape[0]:,}')
        print(f' Number of target contract: {len_contract_target_before:,} → {contract_target.shape[0]:,}')
        print('  Check the duration')
        print(f'  - previous: {contract_previous.계약일자.min()} ~ {contract_previous.계약일자.max()}')
        print(f'  - target: {contract_target.계약일자.min()} ~ {contract_target.계약일자.max()}')

        return contract_previous, contract_target

    def save_product_label(self, contract_previous, contract_target):
        contract_target_names = list(contract_target[self.product_category].value_counts().keys())
        contract_target_codes = list(range(1, len(contract_target_names) + 1))  # 0 is considered as null
        contract_target_label = dict(zip(contract_target_names, contract_target_codes))

        contract_previous_names = pd.Series(contract_previous[self.product_category].value_counts().keys())
        added_contract_previous_names = contract_previous_names[
            ~pd.Series(contract_previous_names).isin(contract_target_names)]
        start_added = max(contract_target_codes)
        added_contract_previous_codes = list(
            range(start_added + 1, start_added + len(added_contract_previous_names) + 1))
        added_contract_previous_label = dict(zip(added_contract_previous_names, added_contract_previous_codes))

        contract_previous_label = {**contract_target_label, **added_contract_previous_label}

        label_dictionary = {'contract_previous_label': contract_previous_label,
                            'contract_target_label': contract_target_label}

        with open('utils/target_label_dictionary.json', 'w') as f:
            json.dump(label_dictionary, f)

    def contract_label_encoding(self, contract_previous, contract_target):
        label_dictionary = read_product_label()
        contract_previous_label = label_dictionary['contract_previous_label']
        contract_target_label = label_dictionary['contract_target_label']

        contract_previous['prdt_cat'] = contract_previous[self.product_category].replace(contract_previous_label)
        contract_target['prdt_cat'] = contract_target[self.product_category].replace(contract_target_label)
        return contract_previous, contract_target

    def revise_ids(self, customer, contract_previous, contract_target):
        target_true_ids = set(contract_target.ID)
        previous_true_ids = set(contract_previous.ID)
        customer_ids = set(customer.ID)
        common_ids = list(target_true_ids.intersection(customer_ids).intersection(previous_true_ids))

        customer_new = customer[customer.ID.isin(common_ids)]
        contract_previous_new = contract_previous[contract_previous.ID.isin(common_ids)]
        contract_target_new = contract_target[contract_target.ID.isin(common_ids)]

        print('■■■ Find common IDs ■■■')
        print(f' Customer set: {len(customer):,} → {len(customer_new):,}')
        print(f' Contract Previous set: {len(contract_previous):,} → {len(contract_previous_new):,}')
        print(f' Contract Target set: {len(contract_target):,} → {len(contract_target_new):,}\n')

        return customer_new, contract_previous_new, contract_target_new

    def build_dataset(self):
        dev_customer_raw, dev_contract_raw = self.load_dataset(self.dev_date)
        dev_contract_previous, dev_contract_target = self.contract_split(self.dev_date, dev_contract_raw)
        dev_customer, dev_contract_previous, dev_contract_target = self.revise_ids(dev_customer_raw,
                                                                                   dev_contract_previous,
                                                                                   dev_contract_target)

        oot_customer_raw, oot_contract_raw = self.load_dataset(self.oot_date)
        oot_contract_previous, oot_contract_target = self.contract_split(self.oot_date, oot_contract_raw)
        oot_customer, oot_contract_previous, oot_contract_target = self.revise_ids(oot_customer_raw,
                                                                                   oot_contract_previous,
                                                                                   oot_contract_target)

        return {'dev': {'customer': dev_customer,
                        'contract': dev_contract_previous,
                        'target': dev_contract_target},
                'oot': {'customer': oot_customer,
                        'contract': oot_contract_previous,
                        'target': oot_contract_target}}


if __name__ == '__main__':
    data_builder = DataBuilder(data_name='del_variable')
    data_builder.build_dataset()