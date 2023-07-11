import os
import pandas as pd
import multiprocessing as mp
from tqdm import tqdm
import glob
from datetime import datetime

# 'goods_id', 'cat_id', 'brandsn'
# 2701f927daa85882f7a62cf173195b79,95b7688a3e6327a73345ce2da3ce6c89,c615a4473c2c23d2db928d49aa6c192c
# 'user_id', 'goods_id', 'is_clk', 'is_like', 'is_addcart', 'is_order', 'expose_start_time', 'dt'
# 8da2ec07d8bf9bfe1e849cb7e7f25e5c,f6e4f43d18157cbdcdc653c6e35f01fb,1,0,0,0,2023-02-03 17:11:07,20230203

item_columns = ['goods_id', 'cat_id', 'brandsn']
user_columns = ['user_id', 'goods_id', 'is_clk', 'is_like', 'is_addcart', 'is_order', 'expose_start_time', 'dt']
new_column_order = ['goods_id', 'cat_id', 'brandsn', 'user_id', 'is_clk', 'is_like', 'is_addcart', 'is_order', 'expose_start_time', 'dt']

item_dtypes = {
    'goods_id': str,
    'cat_id': str,
    'brandsn': str
}

user_dtypes = {
    'user_id': str,
    'goods_id': str,
    'is_clk': int,
    'is_like': int,
    'is_addcart': int,
    'is_order': int,
    'expose_start_time': str,
    'dt': str
}


def custom_date_parser(x):
    try:
        return pd.to_datetime(x)
    except ValueError:
        return datetime.strptime("2019-01-01 00:00:00", "%Y-%m-%d %H:%M:%S")

def read_file(file, columns, dtypes):
    try:
        df = pd.read_csv(file, names=columns, dtype=dtypes, parse_dates=['expose_start_time'], date_parser=custom_date_parser)
        return df
    except Exception as e:
        print(f"Unable to read file {file}: {e}")
        try:
            df = pd.read_csv(file, names=columns, dtype=str)
            return df
        except Exception as e:
            print(f"Unable to read file {file} with dtype=str: {e}")
            return None


def csv_convert(path, process_func):
    pool = mp.Pool(mp.cpu_count())
    
    file_paths = []
    for root, dirs, file_names in os.walk(path):
        file_paths += [os.path.join(root, file_name) for file_name in file_names if file_name.startswith('part')]

    df_list = pool.map(process_func, file_paths)
    pool.close()
    pool.join()

    # Filter out None values
    df_list = [df for df in df_list if df is not None]

    if df_list:
        return pd.concat(df_list)
    else:
        return pd.DataFrame()


def process_item_data(file_path):
    data_frame = read_file(file_path, item_columns, item_dtypes)
    return data_frame


def process_user_data(file_path):
    item_data_frame = pd.read_csv('data/items.csv', names=item_columns, dtype=item_dtypes)
    user_data_frame = read_file(file_path, user_columns, user_dtypes)
        
    # Fill empty values with 0
    user_data_frame['is_clk'] = user_data_frame['is_clk'].fillna(0).astype(int)
    user_data_frame['is_like'] = user_data_frame['is_like'].fillna(0).astype(int)
    user_data_frame['is_addcart'] = user_data_frame['is_addcart'].fillna(0).astype(int)
    user_data_frame['is_order'] = user_data_frame['is_order'].fillna(0).astype(int)
    
    user_data_frame['expose_start_time'] = pd.to_datetime(user_data_frame['expose_start_time'], errors='coerce')
    merged_data = pd.merge(item_data_frame, user_data_frame, on='goods_id')
    merged_data = merged_data.reindex(columns=new_column_order)
    return pd.DataFrame(merged_data)

# get unique values of colums
def get_unique_column_values(file_name, column_names):
    # Read the CSV file
    try:
        df = pd.read_csv(file_name)
    except Exception as e:
        print(f"Unable to read file {file_name}: {e}")
        return
    # Loop through the list of column names
    for column_name in column_names:
        # Get the unique values of the specified column
        unique_values = df[column_name].unique()
        # Create a new DataFrame with the unique values
        unique_df = pd.DataFrame(unique_values, columns=[column_name])
        # Save the new DataFrame to a CSV file
        unique_df_path = os.path.join(os.path.dirname(
            file_name), f"{column_name}_unique.csv")
        unique_df.to_csv(unique_df_path, index=False)


# Process item data first
item_data_frame = csv_convert('data/training/traindata_goodsid', process_item_data)
item_data_frame.to_csv('data/items.csv', index=False)


# Then process user data
user_data_frame = csv_convert('data/training/traindata_user', process_user_data)
user_data_frame.to_csv('data/ui_all.csv', index=False)

get_unique_column_values('data/items.csv', ['goods_id', 'cat_id', 'brandsn'])
get_unique_column_values('data/ui_all.csv', ['user_id'])