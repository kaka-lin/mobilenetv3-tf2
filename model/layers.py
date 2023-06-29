from typing import Callable

import tensorflow as tf


def hardsigmoid(x):
    return tf.nn.relu6(x + 3.0) / 6.0


def hardswish(x):
    return x * tf.nn.relu6(x + 3.0) / 6.0


def indentity(x):
    return x


def get_activation_layer(name: str):
    if name is None:
        return indentity

    activations = {
        "relu": tf.nn.relu,
        "relu6": tf.nn.relu6,
        "leaky_relu": tf.keras.layers.LeakyReLU,
        "hswish": hardswish,
        "hsigmoid": hardsigmoid,
    }
    return activations[name]


class SEBottleneck(tf.keras.layers.Layer):
    """SENet: Squeeze and Excitation

    This function defines a squeeze structure.

    Args:
        reduction:  squeeze_factor
    """
    def __init__(self,
                 reduction: int = 4,
                 name: str = "se_bottle_neck",
                 **kwargs):
        super(SEBottleneck, self).__init__(name=name, **kwargs)

        self.reduction = reduction

    def build(self, input_shape):
        input_channels = input_shape[-1]
        self.se = tf.keras.Sequential(
            [
                tf.keras.layers.GlobalAveragePooling2D(),
                tf.keras.layers.Dense(input_channels // self.reduction, activation="relu"),
                tf.keras.layers.Dense(input_channels, activation="hard_sigmoid"),
                tf.keras.layers.Reshape([1, 1, input_channels]),
            ]
        )

    def call(self, x, training=False, **kwargs):
        return x * self.se(x, training=training)


class MBV3ConvBlock(tf.keras.layers.Layer):
    """Linear Bottlenecks: bneck


    expand (1x1 conv, 其實就是 point-wise conv) + depthwise + pointwise

    Args:
        expand_size: expansion factor
    """
    def __init__(self,
                exp_size: int,
                out_size: int,
                kernel_size: int,
                strides: int = 1,
                act_name: str = None,
                use_se: bool = False,
                name: str = "mbv3_conv_block",
                **kwargs):
        super(MBV3ConvBlock, self).__init__(name=name, **kwargs)

        self.use_residual_connection = strides == 1
        self.activation = get_activation_layer(act_name)

        # Expand
        self.expand_conv = tf.keras.layers.Conv2D(
            exp_size, kernel_size=1, strides=(1, 1), padding="same"
        )
        self.bn1 = tf.keras.layers.BatchNormalization()

        # Depthwise
        self.depthwise_conv = tf.keras.layers.DepthwiseConv2D(
            kernel_size=kernel_size,
            strides=(strides, strides),
            padding="same",
        )
        self.bn2 = tf.keras.layers.BatchNormalization()

        # SE
        self.se = SEBottleneck() if use_se else None

        # PointWise
        self.conv = tf.keras.layers.Conv2D(
            out_size, kernel_size=1, strides=(1, 1), padding="same")
        self.bn3 = tf.keras.layers.BatchNormalization()

    def call(self, x, training=False, **kwargs):
        y = x

        # Expand
        y = self.expand_conv(y, training=training)
        y = self.bn1(y, training=training)
        y = self.activation(y)

        # Depthwise
        y = self.depthwise_conv(y, training=training)
        y = self.bn2(y, training=training)
        y = self.activation(y)

        # se
        if self.se is not None:
            y = self.se(y)

        # Pointwise
        # using Linear activation
        y = self.conv(y, training=training)
        y = self.bn3(y, training=training)

        # Residual
        # if stride == 1 and in_channels == out_channels:
        if self.use_residual_connection and x.shape[-1] == y.shape[-1]:
            y = x + y
        return y
