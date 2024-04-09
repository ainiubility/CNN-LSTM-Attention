# 编译模型
from libs.attention_model import attention_model
import keras
from keras import layers
from keras import models


def compile_model(time_steps, feature_col_num, class_num, total_examples, batch_size):
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

    model.summary()
    return model
