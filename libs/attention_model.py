## 模型创建
import keras
from keras import layers, models, optimizers
from keras import backend as K

SINGLE_ATTENTION_VECTOR = False


def attention_3d_block(inputs):
    # inputs.shape = (batch_size, time_steps, input_dim)
    input_dim = int(inputs.shape[2])
    a = inputs
    # a = Permute((2, 1))(inputs)
    # a = Reshape((input_dim, TIME_STEPS))(a) # this line is not useful. It's just to know which dimension is what.
    a = layers.Dense(input_dim, activation='softmax')(a)
    if SINGLE_ATTENTION_VECTOR:
        a = layers.Lambda(lambda x: K.mean(x, axis=1), name='dim_reduction')(a)
        a = layers.RepeatVector(input_dim)(a)
    a_probs = layers.Permute((1, 2), name='attention_vec')(a)  # 维数转置

    output_attention_mul = layers.concatenate([inputs, a_probs], axis=-1)  # 把两个矩阵拼接
    return output_attention_mul


def attention_model(time_steps: int, input_dims: int, lstm_units: int, output_dim: int) -> models.Model:
    n_denseUnits = 1024
    n_denselayers = 3

    regularizer = keras.regularizers.l2(0.001)
    inputs = layers.Input(shape=(time_steps, input_dims), name='input')
    x = layers.Dense(units=n_denseUnits, name='dense1', activation='relu', kernel_initializer='HeUniform')(inputs)
    x = layers.Conv1D(filters=512, kernel_size=3, padding='same')(x)
    x = layers.MaxPool1D(pool_size=1)(x)
    x = layers.Conv1D(filters=512, kernel_size=3, padding='same')(x)
    x = layers.MaxPool1D(pool_size=1)(x)
    # x = layers.Dropout(0.3)(x)
    for i in range(n_denselayers):
        x = layers.Dense(units=n_denseUnits, activation='relu', kernel_regularizer=regularizer, kernel_initializer='HeUniform', name='Dense_a_' + str(i + 2))(x)
    # lstm_out = Bidirectional(LSTM(lstm_units, activation='relu'), name='bilstm')(x)
    # 对于GPU可以使用CuDNNLSTM
    # x = layers.Bidirectional(layers.LSTM(lstm_units, return_sequences=False, name='lstm1'))(x)
    x = layers.Dropout(0.3, name='dorpot1')(x)
    # x = attention_3d_block(x)
    # x = layers.MultiHeadAttention(num_heads=3,
    #                               key_dim=64,
    #                               value_dim=64,
    #                               name='attention')(x)
    # for i in range(2):
    #     x = layers.Bidirectional(layers.LSTM(lstm_units, return_sequences=True, name='lstm' + str(i + 2)))(x)
    #     x = layers.Dropout(0.3, name='dorpot' + str(i + 2))(x)
    for i in range(n_denselayers):
        x = layers.Dense(units=n_denseUnits, activation='relu', kernel_regularizer=regularizer, kernel_initializer='HeUniform', name='Dense_b_' + str(i + 2))(x)
    # x = layers.Flatten()(x)
    # attention_mul = layers.Embedding(input_dims, output_dim)(attention_mul)
    output = layers.Dense(3, activation='softmax', name='output_')(x)
    model = keras.Model(inputs=inputs, outputs=output, name='model')
    return model
