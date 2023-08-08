import os
import pandas as pd
import multiprocessing as mp
import glob
from datetime import datetime
from sklearn.preprocessing import LabelEncoder
import pickle
import gc
import csv
from tqdm import tqdm


class DataProcessor:
    def __init__(self):
        # 定义列名
        self.goodid_columns = ['goods_id', 'cat_id', 'brandsn']
        self.user_columns = ['user_id', 'goods_id', 'is_clk', 'is_like', 'is_addcart', 'is_order', 'expose_start_time', 'dt']
        self.merge_columns = ['user_id', 'goods_id', 'cat_id', 'brandsn', 'is_clk', 'is_like', 'is_addcart', 'is_order', 'expose_start_time', 'dt']
        self.numencoded_columns = ['user_id', 'goods_id', 'cat_id', 'brandsn', 'is_clk', 'is_like', 'is_addcart', 'is_order']

        # 定义数据类型
        self.goodid_dtypes = {
            'goods_id': str,
            'cat_id': str,
            'brandsn': str
        }

        self.user_dtypes = {
            'user_id': str,
            'goods_id': str,
            'is_clk': int,
            'is_like': int,
            'is_addcart': int,
            'is_order': int,
            'expose_start_time': str,
            'dt': str
        }
        
        self.merge_dtypes = {
            'user_id': str,
            'goods_id': str,
            'cat_id': str,
            'brandsn': str, 
            'is_clk': int,
            'is_like': int, 
            'is_addcart': int,
            'is_order': int, 
            'expose_start_time': str,
            'dt': str
        }
        
        self.numencoded_dtypes = {
            'user_id': str,
            'goods_id': str,
            'cat_id': str,
            'brandsn': str, 
            'is_clk': int,
            'is_like': int, 
            'is_addcart': int,
            'is_order': int
        }
        
        self.behavior_index = {
            'is_clk': 4,
            'is_like': 5,
            'is_addcart': 6,
            'is_order': 7
        }

        # 定义文件路径
        self.goodsid_data_path = "data/training/traindata_goodsid"
        self.item_data_path = 'data/items.csv'
        self.user_data_path = 'data/training/traindata_user'
        self.merge_dataframe_path = 'data/ui_all.csv'

        self.user_id_unique_path = 'data/user_id_unique.csv'
        self.user_encoder_file = 'data/user_encoder.pkl'
        self.user_feature_path = 'data/user_feature.csv'
        self.user_id_field_name = 'user_id'
        self.user_features = None

        self.item_unique_path = 'data/goods_id_unique.csv'
        self.item_encoder_file = 'data/item_encoder.pkl'

        self.cat_id_unique_path = ['data/cat_id_unique.csv']
        self.cat_encoder_file = 'data/cat_encoder.pkl'
        self.cat_feature_file_names = ['data/cat_clk.feature', 'data/cat_like.feature', 'data/cat_order.feature', 'data/cat_addcart.feature']

        self.brandsn_unique_path = ['data/brandsn_unique.csv']
        self.brandsn_encoder_file = 'data/brandsn_encoder.pkl'
        self.brandsn_feature_file_names = ['data/brandsn_clk.feature', 'data/brandsn_like.feature', 'data/brandsn_order.feature', 'data/brandsn_addcart.feature']

        self.numencoded_dataset_file = 'data/numencoded_ui_all.csv'
        self.sample_train_file = 'data/sample_train.txt'
        self.escape_chars = ['\t', '"', '\n']  # The characters that need to be escaped

        # 初始化编码器
        self.cat_encoder = LabelEncoder()
        self.brandsn_encoder = LabelEncoder()
        self.user_encoder = LabelEncoder()
        self.item_encoder = LabelEncoder()

    """自定义日期解析器"""
    @staticmethod
    def parse_date_or_use_default(x):
        try:
            return pd.to_datetime(x)
        except ValueError:
            return datetime.strptime("2019-01-01 00:00:00", "%Y-%m-%d %H:%M:%S")

    """读取CSV文件工具"""
    @staticmethod
    def safe_read_csv(file, names=None, dtype=None, parse_dates=None):
        try:
            df = pd.read_csv(file, names=names, dtype=dtype, parse_dates=parse_dates)
            # 使用tab分隔，先移除所有tab字符
            df = df.applymap(lambda x: x.replace('\t', '') if isinstance(x, str) else x)
            return df
        except Exception as e:
            print(f"Can not read {file}: {e}")
            return pd.DataFrame()  # 返回一个空的DataFrame


    """处理原始商品数据表"""
    def process_data_goodsid(self, file_path):
        data_frame = self.safe_read_csv(file_path, self.goodid_columns, self.goodid_dtypes)
        # 在写入文件之前，检查并处理包含需要转义的字符的字段
        for column in ['goods_id', 'cat_id', 'brandsn']:
            data_frame[column] = data_frame[column].apply(lambda s: ''.join(ch for ch in s if ch not in self.escape_chars))
        return data_frame


    """处理原始用户数据表"""
    def process_data_user(self, file_path):
        item_data_frame = self.safe_read_csv(self.item_data_path, self.goodid_columns, self.goodid_dtypes)
        user_data_frame = self.safe_read_csv(file_path, self.user_columns, self.user_dtypes, parse_dates=['expose_start_time'])
        
        # 在写入文件之前，检查并处理包含需要转义的字符的字段
        for column in ['user_id', 'goods_id']:
            user_data_frame[column] = user_data_frame[column].apply(lambda s: ''.join(ch for ch in s if ch not in self.escape_chars))
        
        # 填充空值为0
        user_data_frame['is_clk'] = user_data_frame['is_clk'].fillna(0).astype(int)
        user_data_frame['is_like'] = user_data_frame['is_like'].fillna(0).astype(int)
        user_data_frame['is_addcart'] = user_data_frame['is_addcart'].fillna(0).astype(int)
        user_data_frame['is_order'] = user_data_frame['is_order'].fillna(0).astype(int)

        merged_data = pd.merge(item_data_frame, user_data_frame, on='goods_id')
        merged_data = merged_data.reindex(columns=self.merge_columns)
        return merged_data


    """多进程处理目录中的所有文件"""
    def process_directory(self, directory_path, process_function):
        file_list = glob.glob(os.path.join(directory_path, "*"))
        # 创建一个进程池
        with mp.Pool(processes=mp.cpu_count()) as pool:
            # 并行处理每个文件
            result_list = pool.map(process_function, file_list)
            return pd.concat(result_list, ignore_index=True)


    """获取唯一列值"""
    @staticmethod
    def get_unique_column_values(file_name, column_names):
        df = DataProcessor.safe_read_csv(file_name)
        # 循环列名列表
        for column_name in column_names:
            # 获取指定列的唯一值
            unique_values = df[column_name].unique()
            # 创建一个新的DataFrame，包含唯一值
            unique_df = pd.DataFrame(unique_values, columns=[column_name])
            # 将新的DataFrame保存为CSV文件
            unique_df_path = os.path.join(os.path.dirname(file_name), f"{column_name}_unique.csv")
            unique_df.to_csv(unique_df_path, index=False, header=None)


    """将merge_date转换成全数量化"""
    def create_numencoded_dataset(self):
        # ui_frame = pd.read_csv(self.merge_dataframe_path, names=self.merge_columns, dtype=self.merge_dtypes)
        # 由于names=self.merge_columns重复添加表头，导致dtype=self.merge_dtypes指定部分不能转换成int所以报错
        ui_frame = pd.read_csv(self.merge_dataframe_path, sep=',', dtype=self.merge_dtypes)

        # 拟合编码器并保存
        self.cat_encoder.fit(ui_frame['cat_id'])
        with open(self.cat_encoder_file, 'wb') as f:
            pickle.dump(self.cat_encoder, f)

        self.brandsn_encoder.fit(ui_frame['brandsn'])
        with open(self.brandsn_encoder_file, 'wb') as f:
            pickle.dump(self.brandsn_encoder, f)
        
        self.user_encoder.fit(ui_frame['user_id'])
        with open(self.user_encoder_file, 'wb') as f:
            pickle.dump(self.user_encoder, f)

        self.item_encoder.fit(ui_frame['goods_id'])
        with open(self.item_encoder_file, 'wb') as f:
            pickle.dump(self.item_encoder, f)
        
        # 使用LabelEncoder转换'user_id', 'goods_id', 'cat_id', 'brandsn'
        ui_frame['user_id'] = self.user_encoder.transform(ui_frame['user_id'])
        ui_frame['goods_id'] = self.item_encoder.transform(ui_frame['goods_id'])
        ui_frame['cat_id'] = self.cat_encoder.transform(ui_frame['cat_id'])
        ui_frame['brandsn'] = self.brandsn_encoder.transform(ui_frame['brandsn'])

        # 目前模型不需要时间信息，先去掉
        ui_frame = ui_frame.drop(columns=['expose_start_time', 'dt'])
        ui_frame.to_csv(self.numencoded_dataset_file, index=False, sep=',', quoting=csv.QUOTE_NONE)
        print('numencoded_dataset prepare' + ' done')
        gc.collect()


    def create_feature(self, numencoded_dataset_path, selected_fields, feature_field, output_file_name):
        data_frame = pd.read_csv(numencoded_dataset_path, header=0, sep=',', dtype=self.numencoded_dtypes)
        # 按选定的字段分组，并计算特征字段的总和
        grouped_data_frame = data_frame.groupby(selected_fields).agg({feature_field: 'sum'}).reset_index()
        
        # 保留特征字段总和大于0的行
        grouped_data_frame = grouped_data_frame[grouped_data_frame[feature_field] > 0]
        grouped_data_frame.to_csv(output_file_name, index=False, sep=',', quoting=csv.QUOTE_NONE)
        print(f'{output_file_name} prepare' + ' done')

    def remove_zero_columns(self, df):
        # 初始化一个列表来保存所有全为零的列的名称
        zero_cols = []

        # 遍历 DataFrame 中的每一列
        for col in df.columns:
            # 检查该列是否全为零
            if (df[col] == 0).all():
                # 如果是，则将其添加到列表中
                zero_cols.append(col)

        # 从 DataFrame 中删除全为零的列
        df = df.drop(columns=zero_cols)
        df.to_csv('data/user_feature_notzero.tsv', header=True, index=False, sep='\t', quoting=csv.QUOTE_NONE)

        # 返回更新后的 DataFrame，全为零的列的名称，以及全为零的列的数量
        return df, zero_cols, len(zero_cols)
    
    
    """以用户对品类和品牌的行为统计计数创建用户特征"""
    def add_user_feature(self,numencoded_dataset_path, ids_name, feature_file_names):
        df = pd.read_csv(numencoded_dataset_path, header=0, sep=',', dtype=self.numencoded_dtypes)
        # 获取指定user_id列的唯一值
        unique_user_id_num = df[self.user_id_field_name].unique()
        # 以唯一值，创建一个新的DataFrame
        user_feature_frame = pd.DataFrame(unique_user_id_num, columns=[self.user_id_field_name])
        
        # 获取指定特征列的唯一值
        unique_ids_num = df[ids_name].unique()
        # 以唯一值行创建一个新列，追加到user_feature_frame后，并初始化为 0
        new_frame = pd.DataFrame(0, index=user_feature_frame.index, columns=unique_ids_num.tolist())
        user_feature_frame = pd.concat([user_feature_frame, new_frame], axis=1)

        # 读取所有的特征文件，并根据文件中的用户 ID 和特征名在 user_feature_frame 中找到对应的位置，将该位置的值更新为特征值
        for feature_file_name in feature_file_names:
            feature_frame = pd.read_csv(feature_file_name, header=0, sep=',')
            feature_frame = feature_frame.groupby([feature_frame.columns[0], feature_frame.columns[1]])[feature_frame.columns[2]].sum().reset_index()
            # 将求和的结果更新到 user_feature_frame 中对应的位置
            for index, row in feature_frame.iterrows():
                user_id = str(row[feature_frame.columns[0]])
                feature_name = str(row[feature_frame.columns[1]])
                feature_value = row[feature_frame.columns[2]]
                if user_id in user_feature_frame[self.user_id_field_name].values and feature_name in user_feature_frame.columns:
                        user_feature_frame.loc[user_feature_frame[self.user_id_field_name] == user_id, feature_name] += feature_value
                else:
                    print(f"Skip user_id: {user_id}, feature_name: {feature_name}")

        # 将处理完的 user_feature_frame 保存到指定的文件中
        user_feature_frame.to_csv(self.user_feature_path, header=True, index=False, sep=',', quoting=csv.QUOTE_NONE)
        print('user_feature prepare' + ' done')
    
        """处理单行数据"""
    def process_line(self, line_behavior):
        line, behavior = line_behavior
        
        # 将行拆分为各个字段
        items = line.strip().split(',')
        user_id = items[0]
        goods_id = items[1]
        cat_id = items[2]
        brandsn = items[3]

        # 获取 behavior值
        behavior_value = items[self.behavior_index[behavior]]
        # 根据'user_id'，匹配用户特征
        user_feature = self.user_features.loc[int(user_id)]
        user_feature_str = ",".join(["dense_feature:" + str(i) for i in user_feature])
        # 创建一个新行，第一列添加"Click:"，后面填入行为类型对应的数值
        final_line = "Click:" + behavior_value + "," + user_feature_str
        # 将'user_id', 'goods_id','brandsn', 'cat_id'添加到新行后面，前面分别加上“1:”，“2:”，“3:”，“4:”
        final_line += ",1:" + user_id + ",2:" + goods_id + ",3:" + brandsn + ",4:" + cat_id

        return final_line


    def read_data(self, filename):
        with open(filename, 'r') as file:
            for line in file:
                yield line


    """创建训练样本，Click:0 + dense_feature: + ... + dense_feature: + 1:user_id + 2:goods_id + 3:brandsn + 4:cat_id"""
    """通过传入的行为类型，来创建对应的训练样本"""
    def create_sample_train(self, behavior):
        self.user_features = pd.read_csv(self.user_feature_path, skiprows=1, header=None)

        # 通过generator读取量化后数据
        data_generator = self.read_data(self.numencoded_dataset_file)

        # 跳过文件头
        next(data_generator)

        # 文件总行数，先硬编码
        total_lines = 7791765

        with open(self.sample_train_file, 'w') as out_file:
            for line in tqdm(data_generator, total=total_lines):
                record = self.process_line((line, behavior))
                if record is not None:
                    out_file.write(record + '\n')
            print('sample_train prepare' + ' done')
        out_file.close()


    def run(self):
        # 处理商品原始数据
        item_data_frame = self.process_directory(self.goodsid_data_path, self.process_data_goodsid)
        item_data_frame.to_csv(self.item_data_path, index=False)
        
        # 处理用户原始数据
        user_data_frame = self.process_directory(self.user_data_path, self.process_data_user)
        user_data_frame.to_csv(self.merge_dataframe_path, index=False)
        
        # 创建数量化数据集
        self.create_numencoded_dataset()

        # 保存'user_id','goods_id', 'cat_id', 'brandsn'
        # self.get_unique_column_values(self.merge_dataframe_path, ['user_id','goods_id', 'cat_id', 'brandsn'])

        features_to_create = [
            (['user_id', 'cat_id'], 'is_clk', 'data/cat_clk.feature'),
            (['user_id', 'cat_id'], 'is_like', 'data/cat_like.feature'),
            (['user_id', 'cat_id'], 'is_order', 'data/cat_order.feature'),
            (['user_id', 'cat_id'], 'is_addcart', 'data/cat_addcart.feature'),
            (['user_id', 'brandsn'], 'is_clk', 'data/brandsn_clk.feature'),
            (['user_id', 'brandsn'], 'is_like', 'data/brandsn_like.feature'),
            (['user_id', 'brandsn'], 'is_addcart', 'data/brandsn_addcart.feature'),
            (['user_id', 'brandsn'], 'is_order', 'data/brandsn_order.feature'),
        ]
        
        # 创建交互特征
        for selected_fields, feature_field, output_file_name in features_to_create:
            self.create_feature(self.numencoded_dataset_file, selected_fields, feature_field, output_file_name)

        # 根据交互记录创建用户画像
        self.add_user_feature(self.numencoded_dataset_file, 'cat_id', self.cat_feature_file_names)
        self.add_user_feature(self.numencoded_dataset_file, 'brandsn', self.brandsn_feature_file_names)

        self.create_sample_train('is_order')


if __name__ == "__main__":
    dp = DataProcessor()
    dp.run()