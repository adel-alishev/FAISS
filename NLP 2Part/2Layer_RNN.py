import numpy as np
import tensorflow as tf
print(tf.__version__)

BATCH_SIZE = 2
SEQ_LEN = 100
EMB_SIZE = 16

x = np.random.rand(BATCH_SIZE, SEQ_LEN, EMB_SIZE).astype(np.float32)
print(x.shape)

H_Size = 32
OUT_Size = 16
rnn = tf.keras.Sequential([
    tf.keras.layers.SimpleRNN(H_Size, activation='relu', return_sequences=True),
    tf.keras.layers.Dense(OUT_Size)
])
y = rnn(x)
print(y.shape)

