import datetime
import os
import tensorflow as tf

output_folder = "./model_output"

#用来保存模型以及我们需要的所有东西
if not os.path.exists(output_folder):
    os.makedirs(output_folder)
save_format = "hdf5"  #或saved_model
if save_format == "hdf5":
    save_path_models = os.path.join(output_folder, "hdf5_models")
    if not os.path.exists(save_path_models):
        os.makedirs(save_path_models)
    model_save_path = os.path.join(save_path_models, "ckpt_epoch{epoch:02d}_val_acc{val_accuracy:.2f}.hdf5")

elif save_format == "saved_model":
    save_path_models = os.path.join(output_folder, "saved_models")
    if not os.path.exists(save_path_models):
        os.makedirs(save_path_models)
    model_save_path = os.path.join(save_path_models, "ckpt_epoch{epoch:02d}_val_acc{val_accuracy:.2f}.ckpt")
#用来保存日志
log_dir = os.path.join(output_folder, 'logs_{}'.format(datetime.datetime.now().strftime("%Y%m%d-%H%M%S")))
if not os.path.exists(log_dir):
    os.makedirs(log_dir)
# print(class_num)
time_steps = 3
# batch_size = 20
batch_size = 200
epochs = 5 + 200 * len(tf.config.list_physical_devices('GPU'))
lr_decay_epochs = 1
