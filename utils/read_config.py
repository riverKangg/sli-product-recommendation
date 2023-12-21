import json
import torch
import pandas as pd
import numpy as np

from utils import *


class readDataConfig(object):
    def __init__(self, model_name="model", input_name="data"):
        with open(f"input/{input_name}_config.json", "rb") as f:
            self.data_config = json.load(f)

    @property
    def info(self):
        print('Dataset name: ', self.data_config['data_name'])
        # print(self.data_config["data"])
        # print(self.data_config["target"])

    @property
    def target(self):
        return self.data_config["target"]["target_label"]

    @property
    def features(self):
        return self.data_config["features"]

    @property
    def featuremap(self):
        return self.data_config['feature_map']

    @property
    def get_target_categories(self):
        return list(read_product_label()['contract_target_label'].values())

    @property
    def get_target_category_featurename(self):
        return self.data_config["target"]["target_product_categroy"]

    @property
    def get_contract_target_label(self):  # label target
        return read_product_label()['contract_target_label']

    @property
    def get_contract_target_label_flipped(self):
        flipped_labels = {v: k for k, v in read_product_label()['contract_target_label'].items()}
        return flipped_labels

    @property
    def get_target_info(self):
        print('category proportion: ', self.data_config['target']['target_category_percent'])

    @property
    def read(self):
        return self.data_config


class readModelConfig(object):
    def __init__(self, model_name="model"):
        with open(f"models/{model_name}_config.json", "rb") as f:
            self.model_config = json.load(f)
        self.model_name = model_name

    @property
    def info(self):
        print('Model name: ', self.model_config["model_name"])

    @property
    def import_model(self):
        model = torch.load(f"./models/{self.model_name}.pt")
        return model

    @property
    def get_feature_names(self):
        return self.model_config['features']['feature_names']

    @property
    def read(self):
        return self.model_config

    @property
    def get_raw_dataset_name(self):
        return self.model_config['rawdata']['raw_dataset_name']

    @property
    def get_data_path(self):
        return self.model_config['data']['data_path']

    @property
    def get_target_label(self):
        return self.model_config['data']['target_label']

    @property
    def get_train_ids(self):
        return self.model_config['data']['train_ids']

    @property
    def get_test_ids(self):
        return self.model_config['data']['test_ids']

    @property
    def get_epochs(self):
        return self.model_config['fit_params']['epochs']

    @property
    def get_clf_model_path(self):
        return self.model_config['add_clf']['clf_model_path']

    @property
    def get_clf_features(self):
        return self.model_config['add_clf']['clf_features']

    @property
    def get_clf_other_model_path(self):
        return self.model_config['add_clf']['other_model_path']

    @property
    def get_clf_multioutput_columns(self):
        return self.model_config['add_clf']['multioutput_columns']

    @property
    def get_clf_drop_category_name(self):
        return self.model_config['add_clf']['drop_category_name']

    @property
    def get_clf_drop_category_label(self):
        return self.model_config['add_clf']['drop_category_label']