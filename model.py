import tensorflow as tf
from tensorflow.keras import layers

def create_model(board_size):
    inputs = tf.keras.Input(shape=(board_size, board_size, 1))
    x = layers.Conv2D(64, 3, activation='relu', padding='same')(inputs)
    x = layers.Conv2D(64, 3, activation='relu', padding='same')(x)
    x = layers.Flatten()(x)
    x = layers.Dense(128, activation='relu')(x)
    outputs = layers.Dense(board_size * board_size, activation='linear')(x)

    model = tf.keras.Model(inputs, outputs)
    model.compile(optimizer='adam', loss='mse')
    return model
