# cnn-lstm-attention tensorflow pandas 分类任务，样本label 设置方法 ，模型训练，模型测试，评价，对数据进行预测的 python代码示例


以下是一个使用TensorFlow、Pandas和Keras构建CNN-LSTM-Attention模型进行文本分类任务的简化示例，包括数据加载、预处理、模型训练、模型测试、评价和预测的全过程。在这个例子中，我们假设数据集已经被读入为Pandas DataFrame，并且文本数据已经进行了预处理（例如，分词、填充、编码为整数序列等）。

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Conv1D, LSTM, Attention, Dense, Flatten
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping

# 假设已有的DataFrame结构如下：
# df = pd.read_csv('data.csv')
# df['text']  # 文本序列
# df['label']  # 类别标签

# 1. 数据预处理
# 假设已经完成了文本预处理，得到了词汇表和编码好的文本序列
# encoded_texts = ...  # 整数编码的文本序列
# labels = df['label'].values
# numeric_labels = LabelEncoder().fit_transform(labels)
# one_hot_labels = to_categorical(numeric_labels, num_classes=num_classes)

# 2. 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(encoded_texts, one_hot_labels, test_size=0.2, random_state=42)

# 3. 对序列进行填充（确保所有序列长度一致）
maxlen = 100  # 假设最大长度为100
X_train_padded = pad_sequences(X_train, maxlen=maxlen)
X_test_padded = pad_sequences(X_test, maxlen=maxlen)

# 4. 构建CNN-LSTM-Attention模型
model = Sequential([
    Embedding(vocab_size, embedding_dim, input_length=maxlen),
    Conv1D(filters=32, kernel_size=3, padding='same', activation='relu'),
    LSTM(units=64),
    Attention(),
    Dense(units=num_classes, activation='softmax')
])

# 5. 编译模型
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# 6. 添加EarlyStopping回调以防止过拟合
early_stopping = EarlyStopping(monitor='val_loss', patience=3)

# 7. 训练模型
history = model.fit(X_train_padded, y_train, validation_data=(X_test_padded, y_test), epochs=10, callbacks=[early_stopping])

# 8. 评估模型
test_loss, test_accuracy = model.evaluate(X_test_padded, y_test)
print(f'Test Loss: {test_loss}, Test Accuracy: {test_accuracy}')

# 9. 对新数据进行预测
new_data_encoded = ...  # 新的编码文本数据
new_data_padded = pad_sequences(new_data_encoded, maxlen=maxlen)
predictions = model.predict(new_data_padded)
predicted_classes = np.argmax(predictions, axis=-1)  # 获取类别预测
```

注意：
- 该示例仅为指导性质，实际应用中需要根据具体的数据集结构和任务需求进行调整。
- 示例中省略了详细的文本预处理步骤，这部分通常涉及分词、词汇表构建和文本编码。
- CNN、LSTM和Attention层的具体参数应根据实际任务调整。
- `vocab_size` 和 `embedding_dim` 分别是词汇表大小和嵌入向量维度，需要根据实际的词汇表构建结果设定。
- 在预测阶段，需要确保新数据也经过相同的预处理流程。