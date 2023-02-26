import numpy as np
import tensorflow as tf
print(tf.__version__)

BATCH_SIZE = 2
SEQ_LEN = 100
EMB_SIZE = 16

x = np.random.rand(BATCH_SIZE, SEQ_LEN, EMB_SIZE).astype(np.float32)
print(x.shape)

#KERAS
H_SIZE = 32
rnn_layer = tf.keras.layers.SimpleRNN(H_SIZE, activation='relu', return_sequences=True)
y = rnn_layer(x)
print(y.shape)

#Custom
class RNNLayer(tf.keras.Model):
    def __init__(self, h_size):
        super().__init__()
        self.h_size = h_size
        self.fcXH = tf.keras.layers.Dense(self.h_size)
        self.fcHH = tf.keras.layers.Dense(self.h_size, use_bias=False)  # биас не нужен, так как он есть в fcXH

    # RNN ячейка
    def RNN_cell(self, x, h):
        h = tf.nn.relu(self.fcXH(x) + self.fcHH(h))
        return h

    def call(self, x_all):
        batch, length, emb_size = x.shape
        h = tf.zeros((batch, self.h_size))
        h_all = []  # список всех получившихся векторов h

        # Цикл по входной последовательности
        for i in range(length):
            h = self.RNN_cell(x_all[:, i, :], h)
            h_all.append(h)

        # склеиваем все ответы и меняем размерности, чтоб получилось (batch, length, h_size)
        h_all = tf.transpose(tf.stack(h_all), [1, 0, 2])
        return h_all
rnn_layer_my = RNNLayer(H_SIZE)
y = rnn_layer_my(x)
print(y.shape)

rnn_layer_my.fcXH.kernel = rnn_layer.weights[0]
rnn_layer_my.fcHH.kernel = rnn_layer.weights[1]
rnn_layer_my.fcXH.bias = rnn_layer.weights[2]

y = rnn_layer(x)
y_my = rnn_layer_my(x)

print(np.max(np.abs(y.numpy() - y_my.numpy())))