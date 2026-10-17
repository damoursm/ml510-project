"""Attention over completed history. Imported after the adapter configures TensorFlow."""
import numpy as np
import tensorflow as tf


@tf.keras.utils.register_keras_serializable(package="ml_lib")
class SinusoidalPositions(tf.keras.layers.Layer):
    """Give attention the position of each observation without learning extra weights."""

    def build(self, input_shape):
        length, width = input_shape[-2:]
        positions = np.arange(length)[:, None]
        rates = np.exp(-np.log(10000.0) * np.arange(0, width, 2) / width)
        angles = positions * rates
        encoding = np.empty((length, width), dtype=np.float32)
        encoding[:, 0::2] = np.sin(angles)
        encoding[:, 1::2] = np.cos(angles)
        self.encoding = tf.constant(encoding[None, :, :])
        super().build(input_shape)

    def call(self, inputs):
        return inputs + tf.cast(self.encoding, inputs.dtype)


def encode_history(history, config):
    """Project, attend, then pool; no target or unobserved hour enters this encoder."""
    layers = tf.keras.layers
    x = layers.Dense(config.model_dim, name="history_projection")(history)
    x = SinusoidalPositions(name="history_positions")(x)
    for number in range(config.num_blocks):
        prefix = f"encoder_{number + 1}"
        normalized = layers.LayerNormalization(epsilon=1e-6, name=prefix + "_attention_norm")(x)
        attended = layers.MultiHeadAttention(
            num_heads=config.num_heads, key_dim=config.model_dim // config.num_heads,
            dropout=config.dropout, name=prefix + "_attention",
        )(normalized, normalized)
        attended = layers.Dropout(config.dropout)(attended)
        x = layers.Add()([x, attended])
        normalized = layers.LayerNormalization(epsilon=1e-6, name=prefix + "_feedforward_norm")(x)
        hidden = layers.Dense(config.feedforward_dim, activation="relu")(normalized)
        hidden = layers.Dropout(config.dropout)(hidden)
        hidden = layers.Dense(config.model_dim)(hidden)
        hidden = layers.Dropout(config.dropout)(hidden)
        x = layers.Add()([x, hidden])
    x = layers.LayerNormalization(epsilon=1e-6, name="encoder_output_norm")(x)
    return layers.GlobalAveragePooling1D(name="history_pool")(x)
