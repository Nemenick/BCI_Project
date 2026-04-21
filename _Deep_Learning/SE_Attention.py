import tensorflow as tf
from tensorflow.keras import layers

# https://openaccess.thecvf.com/content_cvpr_2018/papers/Hu_Squeeze-and-Excitation_Networks_CVPR_2018_paper.pdf

# TODO: need to output weights (ask chatgpt)
# - in test phase
# - in training phase, for inspection

class SEBlock1D(layers.Layer):
    def __init__(self, reduction=1, return_attention=False, **kwargs):
        super().__init__(**kwargs)
        self.reduction = reduction
        self.return_attention = return_attention

    def build(self, input_shape):
        self.channels = input_shape[-1]

        self.gap = layers.GlobalAveragePooling1D()
        self.fc1 = layers.Dense(self.channels // self.reduction, activation='relu')
        self.fc2 = layers.Dense(self.channels, activation='sigmoid')

    def call(self, inputs):
        # Squeeze
        x = self.gap(inputs)  # (batch, channels)

        # Excitation
        x = self.fc1(x)
        attention = self.fc2(x)  # (batch, channels)

        # Store for later inspection
        self.last_attention = attention

        # Reshape for broadcasting
        attention = tf.expand_dims(attention, axis=1)  # (batch, 1, channels)

        # Scale
        output = inputs * attention

        if self.return_attention:
            return output, attention
        return output