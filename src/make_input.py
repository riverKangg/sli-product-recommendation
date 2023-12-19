import warnings

warnings.filterwarnings('ignore')
from datetime import datetime

from utils import *

seed = 42
random.seed(seed)

data_name_ = 'del_variable'

# Load Real data
# db = DataBuilder(data_name=data_name_)
# data_dict = db.build_dataset()

# save distribution
# from src.data_distributor import DataDistributor
# dev_distributor = DataDistributor('dev_customer_dist')
# dev_distributor.save_distributions(data_dict['dev']['dev_customer'])
# dev_distributor = DataDistributor('dev_contract_dist')
# dev_distributor.save_distributions(data_dict['dev']['dev_contract_previous'])
# dev_distributor = DataDistributor('dev_target_dist')
# dev_distributor.save_distributions(data_dict['dev']['dev_contract_target'])


# Generate sample data
dg = DataGenerator('dev_customer_dist', 'dev_contract_dist', 'dev_target_dist')
cust_df, contract_df, target_df = dg.make_vertual_data()
print(cust_df.head())

data_dict = {'dev': {}, 'oot': {}}
data_dict['dev']['dev_customer'] = cust_df
data_dict['dev']['dev_contract_previous'] = contract_df
data_dict['dev']['dev_contract_target'] = target_df

# make modeling input dataset

data_config = {}
data_config['data_name'] = data_name_

# feature
fp = FeatureProcessor(data_dict['dev']['dev_customer'], data_dict['dev']['dev_contract_previous'],
                      data_dict['dev']['dev_contract_target'])
fp.add_feature_last_n_product()
fp.add_feature_previous_contract_describe()
x = fp.make_input_for_modeling()

data_config['data'] = fp.return_data_describe
data_config['features'] = [item for inner_dict in fp.return_field_dictionary.values() for sublist in inner_dict.values()
                           for item in sublist]
data_config['feature_map'] = fp.return_field_dictionary

# target
tp = TargetProcessor(data_dict['dev']['dev_customer'], data_dict['dev']['dev_contract_previous'],
                     data_dict['dev']['dev_contract_target'])
y = tp.make_target_data()

data_config['target'] = tp.return_target_describe

import pickle

# save config
print('■■■ Save config ■■■')
save_data_name = f"{data_config['data_name']}_{data_config['data']['reference_date']}"
with open(f'input/data_{save_data_name}_config.json', 'w') as f:
    json.dump(data_config, f)
print(f' input/data_{save_data_name}_config.json')

# save dataset
print('■■■ Save dataset ■■■')
save_data = x.join(y).reset_index()
with open(f'input/data_{save_data_name}.pkl', 'wb') as f:
    pickle.dump(save_data, f)
print(f' input/data_{save_data_name}.pkl')

# make out-of-time dataset

oot_config = {}
oot_config['data_name'] = data_name_

# feature
from src.feature_processor import FeatureProcessor

fp = FeatureProcessor(data_dict['oot']['oot_customer'], data_dict['oot']['oot_contract_previous'],
                      data_dict['oot']['oot_contract_target'])
fp.add_feature_last_n_product()
fp.add_feature_previous_contract_describe()
x = fp.make_input_for_modeling()

oot_config['data'] = fp.return_data_describe
oot_config['features'] = [item for inner_dict in fp.return_field_dictionary.values() for sublist in inner_dict.values()
                          for item in sublist]
oot_config['feature_map'] = fp.return_field_dictionary

# target
from src.target_processor import TargetProcessor

tp = TargetProcessor(data_dict['oot']['oot_customer'], data_dict['oot']['oot_contract_previous'],
                     data_dict['oot']['oot_contract_target'])
y = tp.make_target_data()

oot_config['target'] = tp.return_target_describe

import pickle

# save config
print('■■■ Save config ■■■')
save_data_name = f"{oot_config['data_name']}_{oot_config['data']['reference_date']}"
with open(f'input/data_{save_data_name}_config.json', 'w') as f:
    json.dump(oot_config, f)
print(f' input/data_{save_data_name}_config.json')

# save dataset
print('■■■ Save dataset ■■■')
save_data = x.join(y).reset_index()
with open(f'input/data_{save_data_name}.pkl', 'wb') as f:
    pickle.dump(save_data, f)
print(f' input/data_{save_data_name}.pkl')
