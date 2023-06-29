import tensorflow as tf
from typing import Tuple

from layers import MBV3ConvBlock, hardsigmoid, hardswish


class MobileNetV3Small(tf.keras.Model):
    """MobileNetV3 Small

    Args:
        input_shape: An integer or tuple/list of 3 integers, shape
                     of input tensor.
        num_classes: Integer, number of classes.
        include_top: if inculde classification layer.
    """
    def __init__(self,
                 input_shape=None,
                 num_classes=1000,
                 include_top=True,
                 name="mobilenetv3",
                 **kwargs):
        super(MobileNetV3Small, self).__init__(name=name, **kwargs)

        # first layer
        self.conv_bn_stem = tf.keras.Sequential(
            [
                tf.keras.layers.Conv2D(
                    16, kernel_size=3, strides=(2, 2), padding="same"
                ),
                tf.keras.layers.BatchNormalization(),
            ]
        )
        self.activation_stem = hardswish

        # Bottleneck layers
        bneck_configs = [
            [16, 16, 3, 2, "relu", True],
            [72, 24, 3, 2, "relu", False],
            [88, 24, 3, 1, "relu", False],
            [96, 40, 5, 2, "hswish", True],
            [240, 40, 5, 1, "hswish", True],
            [240, 40, 5, 1, "hswish", True],
            [120, 48, 5, 1, "hswish", True],
            [144, 48, 5, 1, "hswish", True],
            [288, 96, 5, 2, "hswish", True],
            [576, 96, 5, 1, "hswish", True],
            [576, 96, 5, 1, "hswish", True],
        ]
        self.bnecks = tf.keras.Sequential(
            [MBV3ConvBlock(*config, name=f"{name}_{i:02d}") for i, config in enumerate(bneck_configs)]
        )

        # feat layer
        self.conv_bn_feat = tf.keras.Sequential(
            [
                tf.keras.layers.Conv2D(576, kernel_size=1, strides=(1, 1)),
                tf.keras.layers.BatchNormalization(),
            ]
        )
        self.activation_feat = hardswish

        # classification layer
        self.classifier = tf.keras.Sequential([
            tf.keras.layers.Dense(1280),
            tf.keras.layers.Lambda(hardswish),
            tf.keras.layers.Dense(num_classes),
        ])

    def call(self, x, training=False):
        x = self.conv_bn_stem(x, training=training)
        x = self.activation_stem(x)
        x = self.bnecks(x, training=training)
        x = self.conv_bn_feat(x, training=training)
        x = self.activation_feat(x)
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        x = self.classifier(x, training=training)
        return x


if __name__ == "__main__":
    model = MobileNetV3Small()
    model.build((None, 224, 224, 3))
    print(model.summary(expand_nested=True))
    # print(model.count_params())
    # print(model.count_flops())
    # print(model.count_layers())
    # print(model.count_trainable_params())
    # print(model.count_trainable_layers())
    # print(model.count_non_trainable_params())
    # print(model.count_non_trainable_layers())
    # print(model.count_input_output_tensors())
    # print(model.count_input_output_tensors(include_non_trainable=True))
    # print(model.count_input_output_tensors(include_non_trainable=False))
    # print(model.count_input_output_tensors(include_non_trainable=True, include_input=True))
    # print(model.count_input_output_tensors(include_non_trainable=True, include_input=False))
