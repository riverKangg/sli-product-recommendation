import os
import pickle

from utils import *

# 0. Specify the name of the dataset
data_name_ = 'del_variable'

# 1. Make dataset
#   Choose 1-1 or 1-2
# 1-1. Load Real data
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


# 1-2. Generate sample data
dg = DataGenerator('dev_customer_dist', 'dev_contract_dist', 'dev_target_dist')
cust_df, contract_df, target_df = dg.make_virtual_data()

data_dict = {'dev': {}, 'oot': {}}
data_dict['dev']['dev_customer'] = cust_df
data_dict['dev']['dev_contract_previous'] = contract_df
data_dict['dev']['dev_contract_target'] = target_df

oot_dg = DataGenerator('oot_customer_dist', 'oot_contract_dist', 'oot_target_dist')
oot_cust_df, oot_contract_df, oot_target_df = oot_dg.make_virtual_data()
data_dict['oot']['oot_customer'] = oot_cust_df
data_dict['oot']['oot_contract_previous'] = oot_contract_df
data_dict['oot']['oot_contract_target'] = oot_target_df

# 2. Make development dataset
data_config = {}
data_config['data_name'] = data_name_

# 2-1. Feature
fp = FeatureProcessor(data_dict['dev']['dev_customer'], data_dict['dev']['dev_contract_previous'],
                      data_dict['dev']['dev_contract_target'])
fp.add_feature_last_n_product()
fp.add_feature_previous_contract_describe()
x = fp.make_input_for_modeling()

data_config['data'] = fp.return_data_describe
data_config['features'] = [item for inner_dict in fp.return_field_dictionary.values() for sublist in inner_dict.values()
                           for item in sublist]
data_config['feature_map'] = fp.return_field_dictionary

# 2-2. Target
tp = TargetProcessor(data_dict['dev']['dev_customer'], data_dict['dev']['dev_contract_previous'],
                     data_dict['dev']['dev_contract_target'])
y = tp.make_target_data()

data_config['target'] = tp.return_target_describe

# 2-3. save config
print('■■■ Save config ■■■')
os.makedirs('input', exist_ok=True)
save_data_name = f"{data_config['data_name']}_{data_config['data']['reference_date']}"
with open(f'input/data_{save_data_name}_config.json', 'w') as f:
    json.dump(data_config, f)
print(f' input/data_{save_data_name}_config.json')

# 2-4. save dataset
print('■■■ Save dataset ■■■')
save_data = x.join(y).reset_index()
with open(f'input/data_{save_data_name}.pkl', 'wb') as f:
    pickle.dump(save_data, f)
print(f' input/data_{save_data_name}.pkl')

# 3. Make out-of-time dataset
oot_config = {}
oot_config['data_name'] = data_name_

# 3-1. Feature
fp = FeatureProcessor(data_dict['oot']['oot_customer'], data_dict['oot']['oot_contract_previous'],
                      data_dict['oot']['oot_contract_target'])
fp.add_feature_last_n_product()
fp.add_feature_previous_contract_describe()
x = fp.make_input_for_modeling()

oot_config['data'] = fp.return_data_describe
oot_config['features'] = [item for inner_dict in fp.return_field_dictionary.values() for sublist in inner_dict.values()
                          for item in sublist]
oot_config['feature_map'] = fp.return_field_dictionary

# 3-2. Target
tp = TargetProcessor(data_dict['oot']['oot_customer'], data_dict['oot']['oot_contract_previous'],
                     data_dict['oot']['oot_contract_target'])
y = tp.make_target_data()

oot_config['target'] = tp.return_target_describe

# 3-3. save config
print('■■■ Save config ■■■')
save_data_name = f"{oot_config['data_name']}_{oot_config['data']['reference_date']}"
with open(f'input/data_{save_data_name}_config.json', 'w') as f:
    json.dump(oot_config, f)
print(f' input/data_{save_data_name}_config.json')

# 3-4. save dataset
print('■■■ Save dataset ■■■')
save_data = x.join(y).reset_index()
with open(f'input/data_{save_data_name}.pkl', 'wb') as f:
    pickle.dump(save_data, f)
print(f' input/data_{save_data_name}.pkl')
