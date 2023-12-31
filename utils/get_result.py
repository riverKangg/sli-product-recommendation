import json
import torch
import pickle
import joblib
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
import lightgbm as lgb
from itertools import chain
from dateutil.relativedelta import relativedelta

from utils import *
from utils.read_config import readModelConfig, readDataConfig


class modelResult(object):
    def __init__(self, model_name="model"):
        self.model_name = model_name
        self.model = torch.load(f"./models/{model_name}.pt")

        self.model_config = readModelConfig(model_name)
        self.data_config = readDataConfig(model_name)

        self.model_config.info
        self.data_config.info

        self.flipped_labels = self.data_config.get_contract_target_label_flipped
        self.old_model_categories = ['38.건강', '37.치아', '43.적격연금', '32.암', '20.CI', '41.일반연금', '33.상해', '11.일반종신',
                                     '39.간병', '12.변액종신', '42.변액연금']

    def read_trainset(self):
        data_path = self.model_config.get_data_path
        with open(data_path, 'rb') as f:
            dataset = pickle.load(f)
        return dataset['train']

    def read_testset(self):
        data_path = self.model_config.get_data_path
        with open(data_path, 'rb') as f:
            dataset = pickle.load(f)
        return dataset['test']

    def read_oot_testset(self):
        oot_data_name = self.model_config.get_raw_dataset_name.replace('202305', '202308')
        with open(f'input/{oot_data_name}.pkl', 'rb') as f:
            oot_data = pickle.load(f)
        return oot_data

    def make_target_tamplate(self):
        target_tamplate = pd.DataFrame()

        # label target
        labels = self.data_config.get_contract_target_label
        target_tamplate[self.data_config.get_target_category_featurename] = list(
            map(lambda x: labels[x], self.data_config.get_target_categories))
        return target_tamplate

    def make_model_input(self, data):
        feature_name = self.model_config.get_feature_names
        target = self.model_config.get_target_label

        X = {name: data[name] for name in feature_name}
        if target in data.columns:
            y = data[target].values
            return X, y
        else:
            return X

    def get_person_result_table(self, person_id=None, data=None):
        """
            person_id : int
            data : dataframe, result of make_input.ipynb
        """
        person_id = random.choice(self.model_config.get_test_ids) if person_id is None else person_id
        data = self.read_testset() if data is None else data

        del_cols = ['target_category', 'target', 'has_item']
        person = data[data.ID == person_id].drop(columns=del_cols, errors='ignore').drop_duplicates()
        assert person.shape[0] == 1, "Check <person>"

        person_repedated = pd.concat([person] * self.target_tamplate.shape[0], ignore_index=True)
        if self.target_tamplate.shape[0] != person_repedated.shape[0]:
            raise ValueError("Check <target_tamplate>, <person_repedated>")

        person_rec = pd.concat([person_repedated, self.target_tamplate], axis=1)
        if 'has_item' in self.model_config.get_feature_names:
            target_items = list(person_rec[self.data_config.get_target_category_featurename])
            person_rec['has_item'] = list(map(lambda y: int(person[f'pre_yn_{y}']), target_items))
        person_rec_input = self.make_model_input(person_rec)

        person_rec['prob'] = self.model.predict(person_rec_input)
        person_rec_result = person_rec[['ID', 'prob', 'target_category']]
        person_rec_result['prdct'] = person_rec_result[self.data_config.get_target_category_featurename].replace(
            self.flipped_labels)
        person_rec_result = person_rec_result[['ID', 'prob', 'prdct']].sort_values(by='prob',
                                                                                   ascending=False).reset_index(
            drop=True)
        person_rec_result['result_rank'] = list(map(lambda x: str(x + 1), range(len(person_rec_result))))
        return person_rec_result

    def get_accuracy(self, data=None, old=True, out_of_time=False):
        """
        make result dataframe by ID
            data: dictionary, result of read_testset module
            old: bool, add old model results
            out_of_time: bool, indicates whether the data is out of the time set
        """
        if data is not None:
            testdata = data[:]
        elif out_of_time:
            testdata = self.read_oot_testset()
        else:
            testdata = self.read_testset()

        true_target_data = testdata[testdata[self.model_config.get_target_label] == 1]
        id_list = list(set(true_target_data.ID))

        del_cols = ['target_category', 'target', 'has_item']
        true_target_input = true_target_data.drop(columns=del_cols, errors='ignore').drop_duplicates()
        assert len(set(true_target_data.ID)) == true_target_input.shape[0] == len(id_list)

        # true items by ID
        df_true_items = true_target_data[['ID', 'target_category']]
        df_true_items['true_prdct'] = df_true_items['target_category'].replace(self.flipped_labels)
        df_true_items = df_true_items.groupby('ID')['true_prdct'].apply(list).reset_index(name='true_item').set_index(
            'ID')

        # make test tamplate
        test_tamplate = pd.DataFrame()
        labels = self.data_config.get_contract_target_label
        test_tamplate['ID'] = np.repeat(id_list, len(labels))
        label_codes = list(labels.values())
        test_tamplate['target_category'] = label_codes * len(id_list)
        test_tamplate = test_tamplate.merge(true_target_input, on='ID', how='left')
        assert test_tamplate.shape[0] == len(id_list) * len(label_codes)

        # modeling
        model_input = self.make_model_input(test_tamplate)
        model_pred = self.model.predict(model_input)

        # model result
        df_model_result = test_tamplate[['ID', 'target_category']]
        df_model_result['prob'] = model_pred
        df_model_result['prdct'] = df_model_result['target_category'].replace(self.flipped_labels)

        # best probability by ID
        idx = df_model_result.groupby('ID')['prob'].idxmax()
        df_model_result_best = df_model_result.loc[idx]
        df_model_result_best = df_model_result_best.set_index('ID').drop(columns=['target_category']).rename(
            columns={'prdct': 'pred_item'})
        df_model_result_best = df_model_result_best.join(df_true_items)
        df_model_result_best['accuracy'] = list(
            map(lambda pred, true: 1 if pred in true else 0, df_model_result_best.pred_item,
                df_model_result_best.true_item))

        # data for accuracy: ['pred_item','prob','true_item','accuracy']
        accuracy_df = df_model_result_best[['pred_item', 'prob', 'true_item', 'accuracy']]

        print(f"Num of Testset ids: {len(id_list)}")
        accuracy = round(sum(accuracy_df.accuracy) / len(accuracy_df.accuracy) * 100, 1)
        print(f"Accuracy: {accuracy}%")

        if old:
            reference_date = '202308' if out_of_time else '202305'
            print(f'\nOLD data reference date: {reference_date}')

            # predicted: old model result
            old = pd.read_csv(f'data/OLD_{reference_date}.csv', encoding='euc-kr', converters={'ID': str}).set_index(
                'ID')
            old = old.rename(columns={'OLD': 'old_item'})
            old = old[old.index.isin(id_list)]
            assert old.index.name == 'ID', 'Check if the index of old data is "ID"'
            assert old.index.is_unique, 'old data has duplicated ID'

            # true data for old(all categroies)
            reference_date = '202308' if out_of_time else '202305'
            split_date = pd.to_datetime(reference_date, format='%Y%m') + relativedelta(months=1)

            old_contract = pd.read_csv(f'data/PROD_BASE_{reference_date}.csv', encoding='euc-kr',
                                       converters={'ID': str}).dropna()
            old_contract = old_contract[
                ~old_contract.상품분류.isin(['34.실손', '80.미니보험']) & ~old_contract.상품중분류2.isin(['34.실손', '36.단체', '기타'])]
            old_contract_target = old_contract[pd.to_datetime(old_contract['계약일자'], format='%Y%m%d') >= split_date][
                ['ID', '상품중분류2']].drop_duplicates()
            old_contract_target = old_contract_target[
                old_contract_target.ID.isin(id_list) & old_contract_target.상품중분류2.isin(self.old_model_categories)]
            old_contract_target = old_contract_target.rename(columns={'상품중분류2': 'true_prdct'})
            assert old_contract_target.shape[1] == 2

            # true items for old data by ID
            old_true_items = old_contract_target.groupby('ID')['true_prdct'].apply(list).reset_index(
                name='true_item').set_index('ID')
            assert old_true_items.index.name == 'ID', 'Check if the index of old_true_items is "ID"'

            # data for old accuracy: ['true_item','old_item','old_accuracy']
            old_accuracy_df = old_true_items.join(old)
            old_accuracy_df['old_accuracy'] = list(
                map(lambda old, true: int(old in true), old_accuracy_df['old_item'], old_accuracy_df['true_item']))
            assert ~old_accuracy_df.isna().any().any(), '"old_accuracy_df" contains NaN values'

            accuracy_old = round(sum(old_accuracy_df.old_accuracy) / len(old_accuracy_df.old_accuracy) * 100, 1)
            print(f"Accuracy OLD: {accuracy_old}%")

            return accuracy_df, accuracy, old_accuracy_df, accuracy_old

        return accuracy_df, accuracy

    def return_true_by_category(self, accuracy_df):
        # true data : ['true_cnt', 'true_pct']
        true_lst = list(chain(*accuracy_df.true_item.tolist()))
        df_true = pd.Series(true_lst).value_counts().to_frame()
        df_true.columns = ['true_cnt']
        df_true['true_pct'] = round(df_true['true_cnt'] / sum(df_true['true_cnt']) * 100, 1)
        assert list(df_true.columns) == ['true_cnt', 'true_pct'], 'Check df_true column'
        return df_true

    def get_accuracy_by_category(self, accuracy_df, old_accuracy_df=None):
        df_true = self.return_true_by_category(accuracy_df)

        # accuracy by category for NEW MODEL
        df_category_accuracy = accuracy_df.groupby(['pred_item']).agg({'accuracy': ['count', 'sum']})
        df_category_accuracy.columns = ['pred_cnt', 'pred_true']
        df_category_accuracy['pred_pct'] = round(
            df_category_accuracy['pred_cnt'] / sum(df_category_accuracy['pred_cnt']) * 100, 1)
        df_category_accuracy = df_true.join(df_category_accuracy).fillna(0)
        df_category_accuracy[['pred_cnt', 'pred_true']] = df_category_accuracy[['pred_cnt', 'pred_true']].astype(int)

        df_category_accuracy['accuracy_pred'] = round(
            df_category_accuracy['pred_true'] / df_category_accuracy['pred_cnt'] * 100, 1)
        df_category_accuracy['accuracy_true'] = round(
            df_category_accuracy['pred_true'] / df_category_accuracy['true_cnt'] * 100, 1)
        df_accuracy_by_category = df_category_accuracy[
            ['true_cnt', 'true_pct', 'pred_cnt', 'pred_pct', 'pred_true', 'accuracy_pred', 'accuracy_true']]

        # train&test distribution of categories
        trainset = self.read_trainset()
        trainset = trainset[trainset.target == 1]
        df_train = trainset.target_category.value_counts().to_frame().reset_index()
        df_train.rename(columns={'target_category': 'train_count'}, inplace=True)

        df_train['prod'] = df_train['index'].replace(self.flipped_labels)
        df_train['train_pct'] = round(df_train['train_count'] / sum(df_train['train_count']) * 100, 1)
        df_train = df_train[['prod', 'train_count', 'train_pct']].set_index('prod')

        df_tr_te_distribution = df_train.join(df_true)
        df_tr_te_distribution.columns = ['train_count', 'train_pct', 'test_count', 'test_pct']

        if old_accuracy_df is None:
            return df_accuracy_by_category, df_tr_te_distribution

        else:
            # accuracy by category for OLD MODEL
            df_ture_old = self.return_true_by_category(old_accuracy_df)
            df_category_accuracy_old = old_accuracy_df.groupby(['old_item']).agg(
                {'old_accuracy': ['count', 'sum']}).fillna(0)
            df_category_accuracy_old.columns = ['pred_cnt', 'pred_true']
            df_category_accuracy_old['pred_pct'] = round(
                df_category_accuracy_old['pred_cnt'] / sum(df_category_accuracy_old['pred_cnt']) * 100, 1)

            df_category_accuracy_old = df_ture_old.join(df_category_accuracy_old).fillna(0)
            df_category_accuracy_old['old_accuracy_pred'] = round(
                df_category_accuracy_old['pred_true'] / df_category_accuracy_old['pred_cnt'] * 100, 1)
            df_category_accuracy_old['old_accuracy_true'] = round(
                df_category_accuracy_old['pred_true'] / df_category_accuracy_old['true_cnt'] * 100, 1)
            df_category_accuracy_old[['pred_cnt', 'pred_true']] = df_category_accuracy_old[
                ['pred_cnt', 'pred_true']].astype(int)
            df_accuracy_by_category_old = df_category_accuracy_old[
                ['true_cnt', 'true_pct', 'pred_cnt', 'pred_pct', 'pred_true', 'old_accuracy_pred', 'old_accuracy_true']]

            return df_accuracy_by_category, df_accuracy_by_category_old, df_tr_te_distribution

    def get_accuracy_by_category_add_clf(self, df_accuracy, out_of_time=False, best_model=True):
        if best_model:
            clf_model_path = self.model_config.get_clf_model_path
            clf_model = joblib.load(clf_model_path)
        else:
            other_model_list = self.model_config.get_clf_other_model_path
            idx = input(f'input index\n{other_model_list}')
            clf_model_path = other_model_list[int(idx)]
            clf_model = joblib.load(clf_model_path)

        # split data for deepFM and clf
        df_accuracy_deepfm = df_accuracy[
            df_accuracy.pred_item.apply(lambda s: s in self.model_config.get_clf_drop_category_name)]
        df_accuracy_clf = df_accuracy[
            df_accuracy.pred_item.apply(lambda s: s not in self.model_config.get_clf_drop_category_name)]
        assert len(df_accuracy_deepfm) + len(df_accuracy_clf) == df_accuracy.shape[0]
        assert len(set(df_accuracy_clf.index)) == df_accuracy_clf.shape[0]

        # make input
        testset = self.read_oot_testset() if out_of_time else self.read_testset()
        testset = testset[self.model_config.get_clf_features]
        lgb_ids = list(set(df_accuracy_clf.index))
        df_for_clf = testset[testset.ID.isin(lgb_ids)]  # & ~testset.target_category.isin([1,2])

        assert df_for_clf.duplicated().sum() == 0

        X, y = make_base_input(df_for_clf) if 'base' in clf_model_path else make_chain_input(df_for_clf)
        labels_to_cal = list(map(str, self.model_config.get_clf_drop_category_label))
        y = y.drop(columns=labels_to_cal, errors='ignore')
        assert y.shape[1] == len(self.model_config.get_clf_multioutput_columns)
        y_proba = clf_model.predict_proba(X)
        y_pred_class = get_clf_pred(y, y_proba, output='category')

        df_lgb_result = pd.DataFrame(y.index, columns=['ID'])
        df_lgb_result['prdct'] = y_pred_class

        flipped_labels = self.data_config.get_contract_target_label_flipped
        df_lgb_result['pred_item'] = df_lgb_result['prdct'].replace(flipped_labels)
        df_lgb_result = df_lgb_result.set_index('ID')

        df_accuracy_clf['prob'] = np.nan
        df_accuracy_clf.drop(columns='pred_item', inplace=True)
        df_accuracy_clf = df_accuracy_clf.join(df_lgb_result[['pred_item']])
        df_accuracy_clf['accuracy'] = list(
            map(lambda x, y: 1 if x in y else 0, df_accuracy_clf.pred_item, df_accuracy_clf.true_item))
        df_accuracy_clf = df_accuracy_clf[list(df_accuracy_deepfm.columns)]

        df_accuracy_clf['pred_item'] = df_accuracy_clf['pred_item'].fillna('z')

        df_accuracy2 = pd.concat([df_accuracy_deepfm, df_accuracy_clf])
        accuracy_add_clf = round(sum(df_accuracy2.accuracy) / len(df_accuracy2) * 100, 1)
        print(f'Accuracy add CLF: {accuracy_add_clf}%')
        df_accuracy_by_category_clf, _ = self.get_accuracy_by_category(df_accuracy2)

        return accuracy_add_clf, df_accuracy_by_category_clf


if __name__ == "__main__":
    model_result = modelResult(model)
    # mr.params
    # mr.target_items
    # data = model_result.make_test_dataset()
    # print(data)
