# 编译模型

import keras
from keras import layers
from keras import models

## 模型创建
import tensorflow as tf
import keras
from keras import layers, models, optimizers

SINGLE_ATTENTION_VECTOR = False


def attention_3d_block(inputs):
    # inputs.shape = (batch_size, time_steps, input_dim)
    input_dim = int(inputs.shape[2])
    a = inputs
    # a = Permute((2, 1))(inputs)
    # a = Reshape((input_dim, TIME_STEPS))(a) # this line is not useful. It's just to know which dimension is what.
    a = layers.Dense(input_dim, activation='softmax')(a)
    if SINGLE_ATTENTION_VECTOR:
        a = layers.Lambda(lambda x: tf.reduce_mean(x, axis=1), name='dim_reduction')(a)
        a = layers.RepeatVector(input_dim)(a)
    a_probs = layers.Permute((1, 2), name='attention_vec')(a)  # 维数转置

    output_attention_mul = layers.concatenate([inputs, a_probs], axis=-1)  # 把两个矩阵拼接
    return output_attention_mul


def attention_model(time_steps: int, input_dims: int, lstm_units: int, output_dim: int):  # -> Functional | Any
    n_denseUnits = 1024
    n_denselayers = 3

    regularizer = keras.regularizers.l2(0.001)
    inputs = layers.Input(shape=(time_steps, input_dims), name='input')
    # x = layers.Dense(units=1024, name='dense1', activation='relu', kernel_initializer='HeUniform')(inputs)
    x = layers.Conv1D(filters=512, kernel_size=1, activation='relu', kernel_initializer='HeUniform')(inputs)  # , padding = 'same'
    x = layers.MaxPool1D(pool_size=2)(x)
    # x = layers.Dropout(0.3)(x)
    for i in range(3):
        x = layers.Dense(units=1024, activation='relu', kernel_regularizer=regularizer, kernel_initializer='HeUniform', name='Dense_a_' + str(i + 2))(x)
    # lstm_out = Bidirectional(LSTM(lstm_units, activation='relu'), name='bilstm')(x)
    # 对于GPU可以使用CuDNNLSTM
    x = layers.Bidirectional(layers.LSTM(lstm_units, return_sequences=True, name='lstm1'))(x)
    x = layers.Dropout(0.3, name='dorpot1')(x)
    x = attention_3d_block(x)
    # x = layers.MultiHeadAttention(num_heads=3,
    #                               key_dim=64,
    #                               value_dim=64,
    #                               name='attention')(x)
    # for i in range(2):
    #     x = layers.Bidirectional(layers.LSTM(lstm_units, return_sequences=True, name='lstm' + str(i + 2)))(x)
    #     x = layers.Dropout(0.3, name='dorpot' + str(i + 2))(x)
    for i in range(3):
        x = layers.Dense(units=1024, activation='relu', kernel_regularizer=regularizer, kernel_initializer='HeUniform', name='Dense_b_' + str(i + 2))(x)
    x = layers.Flatten()(x)
    # attention_mul = layers.Embedding(input_dims, output_dim)(attention_mul)
    output = layers.Dense(output_dim, activation=tf.nn.tanh, name='output_')(x)
    model = keras.Model(inputs=[inputs], outputs=output)
    return model


def compile_model(time_steps, feature_col_num, class_num, total_examples, batch_size, print_summary=False):
    # class_num=1
    lstm_units = 64
    model = attention_model(time_steps, feature_col_num, lstm_units, class_num)

    #学习率变化设置，使用指数衰减
    train_steps_per_epoch = int(total_examples // batch_size)
    initial_learning_rate = 0.01
    # lr_schedule = keras.optimizers.schedules.ExponentialDecay(initial_learning_rate,
    #                                                              decay_steps=1*train_steps_per_epoch,
    #                                                             decay_rate=0.96,
    #                                                             staircase=True)#initial_learning_rate*0.96**(step/decay_steps)
    #优化算法
    optimizer = keras.optimizers.SGD(learning_rate=initial_learning_rate, momentum=0.95)
    optimizer = keras.optimizers.Adam(learning_rate=initial_learning_rate)
    #损失函数
    loss = keras.losses.CategoricalCrossentropy(from_logits=False)
    #评价指标
    # metrics=[keras.metrics.SparseCategoricalAccuracy(),loss]
    metrics = ['accuracy', 'categorical_crossentropy']  #, 'categorical_crossentropy'

    # 标签平滑损失函数
    # loss = 'mse'
    #上式第二个参数会返回交叉熵的结果，用loss减去该值就会得到正则化的值（于model.losses和相等），这两种定义方式都可以，下边的会显示名称短一些
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    if print_summary:
        model.summary()

    return model
