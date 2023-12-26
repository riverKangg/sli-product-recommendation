# ■■■■  1. Setting  ■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■
dev_reference_date = '202305'
raw_dataset_name = 'data_del_variable_202305'
model_name = "model_del_variable_mini"

drop_features = [
    '직업대분류', '직업군_관계사공통기준','pre_contract_count', 'pre_contract_month_min', 'pre_contract_month_max',
    '최근계약경과월'
]

# '계약자연령', '계약자성별', '투자성향', '업종1', '업종2', '추정소득',  '외국인여부', 'BP상태코드', '컨설턴트여부', '임직원여부', '관심고객여부', 'VIP등급','우량직종여부',
#                  'F00003', 'F00004', 'F00005', 'F00006', 'F00007', 'F00008','F00009', 'F00010', 'F00011', 'F00012',

# 2. import package
import os
import torch
from sklearn.metrics import log_loss, roc_auc_score
from deepctr_torch.models import DeepFM

# import json
import pickle
import random

import pandas as pd
import warnings

warnings.filterwarnings('ignore')

from utils import *

seed = 42
random.seed(seed)

from sklearn.model_selection import GroupShuffleSplit
from deepctr_torch.inputs import SparseFeat, DenseFeat, get_feature_names

# 3. Load Data
# 3-1. load data
with open(f'input/{raw_dataset_name}.pkl', 'rb') as f:
    input_data = pickle.load(f)
oot_dataste_name = raw_dataset_name.replace('2305', '2308')
with open(f'input/{oot_dataste_name}.pkl', 'rb') as f:
    oot_data = pickle.load(f)

# 3-2. load data config
data_config = readDataConfig(input_name=raw_dataset_name)

target = data_config.target
features = data_config.features
featuremap = data_config.featuremap

# DeepFM

# 4. Make DeepFM Input
fixlen_feature_columns = []

# 4-1. numerical features
numerical_features = [item for list_name, sublist in featuremap['numerical'].items() for item in sublist if
                      item not in drop_features]
fixlen_feature_columns += [DenseFeat(feat, 1) for feat in numerical_features]

# 4-2. categorical features
fixlen_feature_columns += [
    SparseFeat(feat, vocabulary_size=max(input_data[feat])+1, # len(set(input_data[feat])),
               embedding_dim=8, group_name=feature_cat) for
    feature_cat, feature_list in featuremap['categorical'].items()
    for feat in feature_list if feat not in drop_features]

# 4-3. last contracts
# fixlen_feature_columns += [SparseFeat(feat, vocabulary_size=len(set(input_data[feat])), embedding_dim="auto", group_name=feature_cat) for feature_cat, feature_list in featuremap['unencoded'].items() for feat in feature_list]
last_product_len = len(read_product_label()['contract_previous_label']) + 1  # add 0
fixlen_feature_columns += [
    SparseFeat(feat, vocabulary_size=last_product_len, embedding_dim=8, group_name='last_product') for feat in
    featuremap['unencoded']['last_product'] if feat not in drop_features]

# 4-4. target category
target_category_len = len(read_product_label()['contract_target_label']) + 1
fixlen_feature_columns += [
    SparseFeat('target_category', vocabulary_size=target_category_len, embedding_dim=8, group_name='target_category')]

print(f'Num of features: {len(fixlen_feature_columns)}')
print('Features:', *fixlen_feature_columns, sep='\n')

# 5. Data Split
group_splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)

for train_index, test_index in group_splitter.split(input_data[features], groups=input_data.ID):
    train = input_data.iloc[train_index].sample(frac=1, random_state=42)
    test = input_data.iloc[test_index].sample(frac=1, random_state=42)

assert len(set(train['ID']).intersection(set(test['ID']))) == 0, "Error: Overlapping IDs found between train and test."

# 6. Save data
feature_names = get_feature_names(fixlen_feature_columns)
train_model_input = {name: train[name] for name in feature_names}
test_model_input = {name: test[name] for name in feature_names}
oot_model_input = {name: oot_data[name] for name in feature_names}
input_dataset = {'train': train, 'test': test, 'oot': oot_data}

# 7. Train DeepFM
# 7-1. DeepFM setting
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

model_params = {"task": 'binary',
                "l2_reg_embedding": 1e-5,
                "device": device,
                }

model = DeepFM(linear_feature_columns=fixlen_feature_columns,
               dnn_feature_columns=fixlen_feature_columns,
               **model_params)

compile_params = {'optimizer': "adagrad",
                  'loss': "binary_crossentropy",
                  'metrics': ["binary_crossentropy", "auc"],

                  }
model.compile(**compile_params)

fit_params = {"batch_size": 64,
              "epochs": 10,
              "validation_split": 0.2,
              "verbose": 2, }
# 7-2. Train DeepFM
history = model.fit(train_model_input, train[target].values, **fit_params)

# 7-3. DeepFM Result
pred_ans = model.predict(test_model_input)
print("")
print("test LogLoss", round(log_loss(test[target].values, pred_ans), 4))
print("test AUC", round(roc_auc_score(test[target].values, pred_ans), 4))

# 8. Save
model_name = f'{model_name}_{len(fixlen_feature_columns)}'

# 8-1. Save Model
model_path = f"models/{model_name}.pt"
os.makedirs('models/', exist_ok=True)

torch.save(model, model_path)

# 8-2. Save dataset

dataset_path = f'input/{model_name}_dataset.pkl'
with open(dataset_path, 'wb') as f:
    pickle.dump(input_dataset, f)

# 8-3. Save config

model_config_contants = {}
model_config_contants['model_name'] = model_name
model_config_contants['model_path'] = model_path
model_config_contants['random_seed'] = seed

# parameters
model_config_contants["model_params"] = model_params
model_config_contants['compile_params'] = compile_params
model_config_contants["fit_params"] = fit_params

# raw dataset
model_config_contants['rawdata'] = {'raw_dataset_path': f'input/{raw_dataset_name}.pkl',
                                    'raw_dataset_name': raw_dataset_name}
# dataset
model_config_contants['data'] = {'data_path': dataset_path,
                                 'target_label': target,
                                 'num_of_features': len(feature_names),
                                 'num_of_trainset': len(train),
                                 'num_of_testset': len(test),

                                 'train_ids': list(set(train.ID)),
                                 'test_ids': list(set(test.ID))}

model_config_contants['features'] = {'feature_names': get_feature_names(fixlen_feature_columns),
                                     'features': fixlen_feature_columns}

with open(f'models/{model_name}_config.json', 'w') as f:
    json.dump(model_config_contants, f)
