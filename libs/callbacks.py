#模型保存格式默认是saved_model,可以自己定义更改原有类来保存hdf5
import os
import numpy as np
import keras
from libs.config import *


def get_callbacks(initial_learning_rate=0.01):
    ckpt = keras.callbacks.ModelCheckpoint(model_save_path.replace('.hdf5', '.keras'), monitor='val_accuracy', verbose=1, save_best_only=False, save_weights_only=False, save_freq='epoch', mode='auto')
    #当模型训练不符合我们要求时停止训练，连续5个epoch验证集精度没有提高0.001%停
    earlystop = keras.callbacks.EarlyStopping(
        monitor='accuracy',  # 监控的量，val_loss, val_acc, loss, acc
        min_delta=0,  # 监控量的变化量，当大于该值时，认为模型在性能上没有提升
        patience=10,  # 当patience个epoch内，监控量没有提升时，停止训练
        verbose=1,  # 0,1,2,3,4,5,6,7,8,9,10,11,12,1
        mode='auto',  # auto, min, max    monitor='val_accuracy',
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

    callbacks = [ckpt, earlystop, lr, tensorboard, terminate, reduce_lr, csv_logger]  #terminate
    return callbacks
