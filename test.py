import tensorflow as tf
import pandas as pd
import numpy as np


def windowed_dataset(dataset: tf.data.Dataset, window_size=5, shift=1, stride=1) -> tf.data.Dataset:
    windows = dataset.window(window_size, shift=shift, stride=stride, drop_remainder=True, name='op_window')

    def sub_to_batch(features, labels) -> tf.data.Dataset:
        features_batches = features.batch(window_size, drop_remainder=True, name='op_batch_1')
        labels_batches = labels.batch(window_size, drop_remainder=True, name='op_batch_2')

        # 返回特征和标签的批处理结果作为元组
        return tf.data.Dataset.zip((features_batches, labels_batches))

    windows = windows.flat_map(sub_to_batch)
    return windows


# 假设有一个包含多个特征和多标签的DataFrame
df = pd.DataFrame({
    'feature1': np.arange(0, 10),  # 第一个特征列
    'feature2': np.arange(10, 20),  # 第二个特征列
    'label1': np.arange(10, 20),  # 第一个标签列
    'label2': np.arange(0, 10),  # 第二个标签列
    # 更多特征和标签...
})
# print(df['label1'])
# 定义特征列名和标签列名
feature_col_names = ['feature1', 'feature2']
label_col_names = ['label1', 'label2']

# 将DataFrame转换为NumPy数组
features = df[feature_col_names].values
labels = df[label_col_names].values

dataset_features = tf.data.Dataset.from_tensor_slices(features)
dataset_labels = tf.data.Dataset.from_tensor_slices(labels)
# 创建数据集
dataset = tf.data.Dataset.from_tensor_slices((features, labels))

# 如果需要对特征或标签做预处理（如归一化、标准化等），可以添加map操作
# dataset = dataset.map(lambda feat, lab: (preprocess_features(feat), preprocess_labels(lab)))

# 对于时间序列数据或其他需要窗口滑动的情况，可以使用window方法并应用batch
# window_size 表示每个窗口包含的样本数
window_size = 2
shift = 1
stride = 1


def make_windowed_dataset(ds: tf.data.Dataset, window_size=window_size, shift=shift, stride=stride) -> tf.data.Dataset:
    windows = ds.window(window_size, shift=shift, stride=stride, drop_remainder=True, name='op_window')

    def sub_to_batch(sub):
        return sub.batch(window_size, drop_remainder=True, name='op_sub_to batch')

    windows = windows.flat_map(sub_to_batch)
    return windows


# 应用窗口滑动
dataset = windowed_dataset(dataset)

# 设置批处理大小
batch_size = 32
# dataset = dataset.batch(batch_size)

# 验证数据集形状
for feat, lab in dataset:
    print('Features shape:', feat.shape)  # (batch_size, window_size, num_features)
    print('Labels shape:', lab.shape)  # (batch_size, window_size, num_labels)
