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


def attention_model(time_steps: int, input_dims: int, lstm_units: int, output_dim: int) -> keras.Model:
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
        x = layers.Dense(units=1024, activation=tf.nn.relu6, kernel_regularizer=regularizer, kernel_initializer='HeUniform', name='Dense_b_' + str(i + 2))(x)
    x = layers.Flatten()(x)
    # attention_mul = layers.Embedding(input_dims, output_dim)(attention_mul)
    output = layers.Dense(output_dim, activation=tf.nn.softmax, name='output_')(x)
    model = keras.Model(inputs=[inputs], outputs=output)
    return model


def compile_model(time_steps, feature_col_num, class_num, print_summary=False):
    # class_num=1
    lstm_units = 64
    model = attention_model(time_steps, feature_col_num, lstm_units, class_num)

    #学习率变化设置，使用指数衰减
    # train_steps_per_epoch = int(total_examples // batch_size)
    initial_learning_rate = 0.01
    # lr_schedule = keras.optimizers.schedules.ExponentialDecay(initial_learning_rate,
    #                                                              decay_steps=1*train_steps_per_epoch,
    #                                                             decay_rate=0.96,
    #                                                             staircase=True)#initial_learning_rate*0.96**(step/decay_steps)
    #优化算法
    optimizer = keras.optimizers.SGD(learning_rate=initial_learning_rate, momentum=0.95)
    optimizer = keras.optimizers.Adam(learning_rate=initial_learning_rate)
    # optimizer = keras.optimizers.RMSprop(learning_rate=initial_learning_rate)
    #损失函数
    loss = keras.losses.CategoricalCrossentropy(from_logits=False)
    loss = keras.losses.categorical_crossentropy
    #评价指标
    # metrics=[keras.metrics.SparseCategoricalAccuracy(),loss]
    metrics = [keras.metrics.mean_absolute_percentage_error, keras.metrics.categorical_accuracy, keras.metrics.LogCoshError()]  #, 'categorical_crossentropy'

    # 标签平滑损失函数
    # loss = 'mse'
    #上式第二个参数会返回交叉熵的结果，用loss减去该值就会得到正则化的值（于model.losses和相等），这两种定义方式都可以，下边的会显示名称短一些
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    if print_summary:
        model.summary()

    return model


#模型保存格式默认是saved_model,可以自己定义更改原有类来保存hdf5
import os
import numpy as np
import keras
from libs.config import *


def get_callbacks(initial_learning_rate=0.01):
    ckpt = keras.callbacks.ModelCheckpoint(model_save_path.replace('.hdf5', '.keras'), monitor='accuracy', verbose=1, save_best_only=False, save_weights_only=False, save_freq='epoch', mode='auto')
    #当模型训练不符合我们要求时停止训练，连续5个epoch验证集精度没有提高0.001%停
    earlystop = keras.callbacks.EarlyStopping(
        monitor='categorical_accuracy',  # 监控的量，val_loss, val_acc, loss, acc
        min_delta=0,  # 监控量的变化量，当大于该值时，认为模型在性能上没有提升
        patience=10,  # 当patience个epoch内，监控量没有提升时，停止训练
        verbose=1,  # 0,1,2,3,4,5,6,7,8,9,10,11,12,1
        mode='max',  # auto, min, max    monitor='val_accuracy',
        baseline=None,  # 基准线，当monitor达到baseline时，patience计数器清零
        restore_best_weights=True,  # 是否恢复训练时验证集上表现最好的权重

    )

    #3、自定义学习率按需衰减，并把整个学习率变化过程保存
    class LearningRateExponentialDecay:

        def __init__(self, initial_learning_rate, decay_epochs, decay_rate):
            self.initial_learning_rate = initial_learning_rate
            self.decay_epochs = decay_epochs
            self.decay_rate = decay_rate

        def __call__(self, epoch):
            dtype = type(self.initial_learning_rate)
            decay_epochs = np.array(self.decay_epochs).astype(dtype)
            decay_rate = np.array(self.decay_rate).astype(dtype)
            epoch = np.array(epoch).astype(dtype)
            p = epoch / decay_epochs
            lr = self.initial_learning_rate * np.power(decay_rate, p)
            return lr

    lr_schedule = LearningRateExponentialDecay(initial_learning_rate, lr_decay_epochs, 0.96)
    lr = keras.callbacks.LearningRateScheduler(lr_schedule, verbose=1)

    Reduce = keras.callbacks.ReduceLROnPlateau(monitor='val_accuracy', factor=0.0001, patience=1, verbose=1, mode='auto', min_delta=0.0001, cooldown=0, min_lr=0)
    #使用tensorboard
    #定义当loss出现nan或inf时停止训练的callback
    terminate = keras.callbacks.TerminateOnNaN()

    #模型损失长时间不除时大程度降低学习率
    # 这个策略通常不于学习率衰减schedule同时使用，或者使用时要合理
    #降低学习率（要比学习率自动周期变化有更大变化和更长时间监控）
    # 模型损失长时间不除时大程度降低学习率
    # 这个策略通常不于学习率衰减schedule同时使用，或者使用时要合理
    reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, verbose=1, min_delta=0.0001, min_lr=0)

    #保存训练过程中大数标量指标，与tensorboard同一个文件
    csv_logger = keras.callbacks.CSVLogger(os.path.join(log_dir, 'logs.log'), separator=',')

    #还要加入tensorboard的使用,这种方法记录的内容有限
    #各个参数的作用请参看文档，需要正确使用
    tensorboard = keras.callbacks.TensorBoard(
        log_dir=log_dir,
        histogram_freq=1,  #对参数和激活做直方图，一定要有测试集
        write_graph=True,  #模型结构图
        write_images=True,  #把模型参数做为图片形式存到
        update_freq='epoch',  #epoch,batch,整数，太频的话会减慢速度
        profile_batch=2,  #记录模型性能
        embeddings_freq=1,
        embeddings_metadata=None  #这个还不太清楚如何使用
    )
    import pathlib

    current_directory = pathlib.Path.cwd()
    print(f'tensorboard --logdir="{pathlib.Path.joinpath(current_directory,log_dir)}" --host=127.0.0.1')

    callbacks = [terminate,  earlystop]  #terminate   ckpt,earlystop,csv_logger,tensorboard,
    return callbacks
