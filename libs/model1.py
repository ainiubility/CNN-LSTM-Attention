## 模型创建
import tensorflow as tf
import keras
from keras import layers, models, optimizers


def compile_model(time_steps: int, input_dims: int, lstm_units: int, output_dim: int) -> keras.Model:
    regularizer = keras.regularizers.l2(0.1)
    model = keras.Sequential([
        layers.Input(shape=(time_steps, input_dims)),
        layers.Conv1D(
            filters=512,
            kernel_size=7,
            padding="same",
            strides=2,
            activation="relu",
            kernel_regularizer=regularizer,
        ),
        layers.Dropout(rate=0.2),
        layers.Conv1D(
            filters=512,
            kernel_size=7,
            padding="same",
            strides=2,
            activation="relu",
            kernel_regularizer=regularizer,
        ),
        layers.Conv1DTranspose(
            filters=512,
            kernel_size=7,
            padding="same",
            strides=2,
            activation="relu",
            kernel_regularizer=regularizer,
        ),
        layers.Dropout(rate=0.2),
        layers.Conv1DTranspose(
            filters=512,
            kernel_size=7,
            padding="same",
            strides=2,
            activation="relu",
            kernel_regularizer=regularizer,
        ),
        layers.Conv1DTranspose(filters=32, kernel_size=7, padding="same"),
        # layers.Flatten(),
        layers.Dense(output_dim, kernel_regularizer=regularizer, activation="softmax")
    ])
    metrics = [keras.metrics.Accuracy(), keras.metrics.mean_absolute_percentage_error, keras.metrics.categorical_accuracy, keras.metrics.LogCoshError()]  #, 'categorical_crossentropy'

    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001), loss=keras.losses.CategoricalCrossentropy(), metrics=metrics)
    model.summary()
    return model
