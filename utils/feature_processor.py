import random
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

from utils import *

seed = 42
random.seed(seed)


class FeatureProcessor(object):
    def __init__(self,
                 customer_dataset,
                 contract_previous_dataset,
                 contract_target_dataset,
                 development_reference_date='202305',
                 ):
        print(f'\n■■■ Feature Processing ■■■')
        self.customer_dataset = customer_dataset if customer_dataset.index.name == 'ID' else customer_dataset.set_index(
            'ID')

        assert len(set(customer_dataset.마감년월)) == 1
        self.yyyymm = list(set(customer_dataset.마감년월))[0]
        print(f' reference date: {self.yyyymm}')
        self.is_development = True if str(self.yyyymm) == development_reference_date else False
        print(f' is_development: {self.is_development}')

        assert self.customer_dataset.index.is_unique
        self.ids = self.customer_dataset[[]]
        self.num_of_ids = self.customer_dataset.shape[0]

        self.contract_previous_dataset = contract_previous_dataset

        self.field_dictionary = self.make_base_field_dictionary()

    def make_base_field_dictionary(self):
        field_dictionary = {'categorical': {}, 'numerical': {}, 'unencoded': {}}
        field_dictionary['categorical']['customer_categroy'] = ['계약자성별', '외국인여부', 'BP상태코드', '컨설턴트여부', '임직원여부', '관심고객여부',
                                                                'VIP등급', '우량직종여부', '직업대분류', '직업군_관계사공통기준', '투자성향',
                                                                '업종1', '업종2']
        field_dictionary['numerical']['customer_numeric'] = ['계약자연령', '추정소득', '최근계약경과월']
        field_dictionary['numerical']['fire_contract'] = ['F00003', 'F00004', 'F00005', 'F00006', 'F00007', 'F00008',
                                                          'F00009', 'F00010', 'F00011', 'F00012']
        return field_dictionary

    def add_field_dictionary(self, data_type, feature_list_name, feature_list):
        if data_type not in ['categorical', 'numerical', 'unencoded']:
            raise ValueError("Invalid data_type. Accepted values are 'categorical', 'numerical' or 'unencoded'")
        self.field_dictionary[data_type][feature_list_name] = feature_list

        # make features

    def add_feature_last_n_product(self, n=5):
        last_product_order = self.contract_previous_dataset.sort_values(by=['ID', '계약일자'], ascending=[True, False])
        last_product_order = last_product_order.groupby(['ID']).head(n)
        last_product_order['order'] = last_product_order.groupby(['ID']).cumcount() + 1

        last_product_category = last_product_order.pivot_table(index='ID', columns='order', values='prdt_cat')
        last_product_category.columns = list(map(lambda x: f'product_{x + 1}', range(n)))

        assert not (last_product_category == 0).any().any()
        last_product_category = last_product_category.fillna(0)  # 0 is considered as null

        # added field dictionary
        self.field_dictionary['unencoded']['last_product'] = list(last_product_category.columns)
        # added to customer data set
        self.customer_dataset = self.customer_dataset.join(last_product_category)
        assert self.customer_dataset.shape[0] == self.num_of_ids
        print(f' - Added features for the last {n} contracted products')

    def add_feature_previous_contract_describe(self):
        data = self.contract_previous_dataset
        data['계약여부'] = 1
        end_date = pd.to_datetime(self.yyyymm, format='%Y%m').to_pydatetime().date()
        data['계약경과월'] = list(
            map(lambda x: relativedelta(end_date, x).years * 12 + relativedelta(end_date, x).months, data['계약일자']))
        data = data[['ID', '계약여부', '계약경과월']]

        data_previous_contract_describe = self.ids
        result_df = data.groupby('ID').agg(pre_contract_count=('계약여부', 'count'),
                                           pre_contract_month_min=('계약경과월', 'min'),
                                           pre_contract_month_max=('계약경과월', 'max'))
        data_previous_contract_describe = data_previous_contract_describe.join(result_df)

        assert len(data_previous_contract_describe[data_previous_contract_describe.pre_contract_count == 0]) == 0
        assert ~data_previous_contract_describe.isnull().values.any()

        self.field_dictionary['numerical']['previoius_contract_describe'] = list(
            data_previous_contract_describe.columns)
        self.customer_dataset = self.customer_dataset.join(data_previous_contract_describe)
        assert self.customer_dataset.shape[0] == self.num_of_ids
        print(' - Added descriptive statistics for features related to previous contracts')

    def add_feature_contract_yn(self):
        data = self.contract_previous_dataset
        data['계약여부'] = 1

        data_prodpre_yn = data.pivot_table(index='ID', columns='prdt_cat', values='계약여부', aggfunc='max').fillna(0)
        data_prodpre_yn.columns = list(map(lambda x: f'pre_yn_{x}', data_prodpre_yn.columns))

        data_prodpre_yn = data_prodpre_yn.fillna(0)

        self.field_dictionary['numerical']['previous_contract_yn'] = list(data_prodpre_yn.columns)
        self.customer_dataset = self.customer_dataset.join(data_prodpre_yn)
        assert self.customer_dataset.shape[0] == self.num_of_ids
        print(' - Added features representing the minimum months elapsed for contracts within each product category')

    def add_feature_contract_month(self):
        data = self.contract_previous_dataset
        end_date = pd.to_datetime(self.yyyymm, format='%Y%m').to_pydatetime().date()
        data['계약경과월'] = list(
            map(lambda x: relativedelta(end_date, x).years * 12 + relativedelta(end_date, x).months, data['계약일자']))

        data_prodpre_month = data.pivot_table(index='ID', columns='prdt_cat', values='계약경과월',
                                              aggfunc='min')
        data_prodpre_month.columns = list(map(lambda x: f'pre_m_{x}', data_prodpre_month.columns))

        data_prodpre_month = data_prodpre_month.fillna(-1)

        self.field_dictionary['nonencoded']['previous_contract_month'] = list(
            data_prodpre_month.columns)  # due to null values
        self.customer_dataset = self.customer_dataset.join(data_prodpre_month)
        assert self.customer_dataset.shape[0] == self.num_of_ids
        print(' - Added features indicating the presence of contracts for each product category')

    def fillna(self, data):
        print('■■■ Handling Null Value ■■■')
        self.feature_with_null = data.columns[data.isnull().any()].tolist()
        print(f' na cols: {self.feature_with_null}')

        categorical_features = [item for sublist in self.field_dictionary['categorical'].values() for item in sublist]
        data[categorical_features] = data[categorical_features].fillna('unknown')
        return data

    def encoding(self, make_new_encoder):
        all_features = [item for inner_dict in self.field_dictionary.values() for sublist in inner_dict.values() for
                        item in sublist]
        data = self.fillna(self.customer_dataset[all_features])

        print('■■■ Feature Encoding ■■■')
        categorical_features = [item for sublist in self.field_dictionary['categorical'].values() for item in sublist]
        numeric_features = [item for sublist in self.field_dictionary['numerical'].values() for item in sublist]
        print(f' categorical features: {len(categorical_features), categorical_features}')
        print(f' numerical features: {len(numeric_features), numeric_features}')
        if make_new_encoder:
            os.makedirs('utils/encoders/', exist_ok=True)
            for feat in categorical_features:
                lbe = LabelEncoder()
                lbe_data = list(data[feat]) + ['unknown']
                lbe.fit(lbe_data)
                joblib.dump(lbe, f'utils/encoders/{feat}_label_encoder.joblib')
                del lbe

            for feat in numeric_features:
                mms = MinMaxScaler(feature_range=(0, 1))
                mms.fit(data[[feat]])
                joblib.dump(mms, f'utils/encoders/{feat}_minmax_scaler.joblib')
                del mms

        for feat in categorical_features:
            filename = f'utils/encoders/{feat}_label_encoder.joblib'
            lbe = joblib.load(filename)
            data[feat] = data[feat].astype('str')
            data[feat] = lbe.transform(data[feat].map(lambda s: s if s in lbe.classes_ else 'unknown'))
            del lbe

        for feat in numeric_features:
            mms = joblib.load(f'utils/encoders/{feat}_minmax_scaler.joblib')
            data[feat] = mms.transform(data[[feat]])
            del mms

        self.customer_dataset = data

    def make_input_for_modeling(self):
        if self.is_development:
            self.encoding(make_new_encoder=True)
        else:
            self.encoding(make_new_encoder=False)

        assert self.customer_dataset.shape[0] == self.num_of_ids
        return self.customer_dataset

    @property
    def return_field_dictionary(self):
        return self.field_dictionary

    @property
    def return_data_describe(self):
        data_describe = {'reference_date': self.yyyymm,
                         'is_development': self.is_development,
                         'num_of_ids': self.num_of_ids,
                         'feature_with_null': self.feature_with_null,
                         }
        return data_describe


if __name__ == '__main__':
    dg = DataGenerator('dev_customer_dist', 'dev_contract_dist', 'dev_target_dist')
    cust_df, contract_df, target_df = dg.make_virtual_data()

    data_dict = {'dev': {}, 'oot': {}}
    data_dict['dev']['dev_customer'] = cust_df
    data_dict['dev']['dev_contract_previous'] = contract_df
    data_dict['dev']['dev_contract_target'] = target_df

    fp = FeatureProcessor(data_dict['dev']['dev_customer'], data_dict['dev']['dev_contract_previous'],
                          data_dict['dev']['dev_contract_target'])
    fp.add_feature_last_n_product()
    fp.add_feature_previous_contract_describe()
    cust_data = fp.make_input_for_modeling()
