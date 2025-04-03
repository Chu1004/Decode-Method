import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers, regularizers
import matplotlib.pyplot as plt

# 信噪比範圍
SNR_dB = np.arange(0, 10, 1)
SNR_T = 10**(6/10)
R = 16 / 7  # 信號碼率
noise_stddev = tf.sqrt(1 / (2 * R * SNR_T))

# DNN Model
def model_compile(SNR):
    model = keras.models.Sequential([
        keras.layers.Dense(64, activation='sigmoid', input_shape=(15,),
                           kernel_regularizer=regularizers.l2(0.001)),
        keras.layers.BatchNormalization(),
        keras.layers.Dropout(0.2),

        keras.layers.Dense(128, activation='swish',
                           kernel_regularizer=regularizers.l2(0.001)),
        keras.layers.BatchNormalization(),
        keras.layers.Dropout(0.3),

        keras.layers.Dense(256, activation='swish',
                           kernel_regularizer=regularizers.l2(0.001)),
        keras.layers.BatchNormalization(),
        keras.layers.Dropout(0.3),

        keras.layers.Dense(512, activation='swish',
                           kernel_regularizer=regularizers.l2(0.001)),
        keras.layers.BatchNormalization(),
        keras.layers.Dropout(0.4),

        keras.layers.Dense(2048, activation='softmax')
    ])

    opti_fn = keras.optimizers.Adam(
        learning_rate=0.001,
        beta_1=0.9,
        beta_2=0.999
    )

    loss_fn = keras.losses.CategoricalCrossentropy(
        label_smoothing=0.1
    )

    model.compile(
        optimizer=opti_fn,
        loss=loss_fn,
        metrics=['accuracy']
    )

    return model, 100

# Auto-Encoder
input_bits = keras.Input(shape=(16,))

x = layers.Dense(32, kernel_regularizer=keras.regularizers.l2(0.0005))(input_bits)
x = layers.BatchNormalization()(x)
x = layers.Activation('swish')(x)

x = layers.Dense(20, kernel_regularizer=keras.regularizers.l2(0.0005))(x)
x = layers.BatchNormalization()(x)
x = layers.Activation('swish')(x)

x = layers.Dropout(0.2)(x)

x = layers.Dense(14, kernel_regularizer=keras.regularizers.l2(0.0005))(x)
x = layers.BatchNormalization()(x)
x = layers.Activation('swish')(x)

x = layers.Dense(7, kernel_regularizer=keras.regularizers.l2(0.0005))(x)
x = layers.BatchNormalization()(x)

encoded = layers.Activation('linear')(x)
encoded = layers.Lambda(lambda x: x / (tf.norm(x, ord=2, axis=-1, keepdims=True) + 1e-8))(encoded)

noise = layers.GaussianNoise(noise_stddev)(encoded)

x = layers.Dense(7, kernel_regularizer=keras.regularizers.l2(0.0005))(noise)
x = layers.BatchNormalization()(x)
x = layers.Activation('leaky_relu')(x)

x = layers.Dense(20, kernel_regularizer=keras.regularizers.l2(0.0005))(x)
x = layers.BatchNormalization()(x)
x = layers.Activation('leaky_relu')(x)

x = layers.Dense(28, kernel_regularizer=keras.regularizers.l2(0.0005))(x)
x = layers.BatchNormalization()(x)
x = layers.Activation('leaky_relu')(x)

x = layers.Dropout(0.2)(x)

x = layers.Dense(32, kernel_regularizer=keras.regularizers.l2(0.0005))(x)
x = layers.BatchNormalization()(x)
x = layers.Activation('leaky_relu')(x)

decoded = layers.Dense(16, activation='sigmoid')(x)

# Deep Learning Model
def deep_learning_model():
    model = keras.Sequential([
        layers.Dense(128, activation='relu', input_shape=(7,)),
        layers.BatchNormalization(),
        layers.Dense(64, activation='relu'),
        layers.BatchNormalization(),
        layers.Dense(64, activation='relu'),
        layers.BatchNormalization(),
        layers.Dense(16, activation='softmax')
    ])

    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

# ML Decoding
def ML_decode(y):
    G = np.array([[1, 0, 0, 0, 1, 1, 0],
                  [0, 1, 0, 0, 1, 0, 1],
                  [0, 0, 1, 0, 0, 1, 1],
                  [0, 0, 0, 1, 1, 1, 1]])
    messages = np.array([[m0, m1, m2, m3]
                         for m0 in range(2)
                         for m1 in range(2)
                         for m2 in range(2)
                         for m3 in range(2)])
    codeword = messages @ G % 2
    codeword = codeword * 2 - 1
    y = y[:, np.newaxis, :]
    distances = np.linalg.norm(y - codeword, axis=2)
    min_indices = np.argmin(distances, axis=1)
    m_hat = messages[min_indices]
    return m_hat

# Syndrome Decoding
def syndrome_decode(d_hat):
    H = np.array([[1, 1, 0, 1, 1, 0, 0],
                  [1, 0, 1, 1, 0, 1, 0],
                  [0, 1, 1, 1, 0, 0, 1]])
    s = np.mod(d_hat @ H.T, 2)
    error_map = {
        (0, 0, 0): np.array([0, 0, 0, 0, 0, 0, 0]),
        (0, 0, 1): np.array([0, 0, 0, 0, 0, 0, 1]),
        (0, 1, 0): np.array([0, 0, 0, 0, 0, 1, 0]),
        (0, 1, 1): np.array([0, 0, 1, 0, 0, 0, 0]),
        (1, 0, 0): np.array([0, 0, 0, 0, 1, 0, 0]),
        (1, 0, 1): np.array([0, 1, 0, 0, 0, 0, 0]),
        (1, 1, 0): np.array([1, 0, 0, 0, 0, 0, 0]),
        (1, 1, 1): np.array([0, 0, 0, 1, 0, 0, 0]),
    }
    error = np.array([error_map.get(tuple(row), np.zeros(7, dtype=int)) for row in s])
    x_hat = np.mod(d_hat + error, 2)
    m_hat = x_hat[:, :4]
    return m_hat
