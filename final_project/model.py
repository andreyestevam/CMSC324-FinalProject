import tensorflow as tf
import torch

def build_unet3d(input_shape=(64, 64, 64, 4), base_filters=16, dropout_rate=0.20):

    inputs = tf.keras.Input(shape=input_shape)

    # Encoder
    c1 = conv3d_block(inputs, base_filters)
    p1 = tf.keras.layers.MaxPooling3D(2)(c1)           # 64→32

    c2 = conv3d_block(p1, base_filters * 2)
    p2 = tf.keras.layers.MaxPooling3D(2)(c2)           # 32→16

    c3 = conv3d_block(p2, base_filters * 4)
    p3 = tf.keras.layers.MaxPooling3D(2)(c3)           # 16→8

    # Bottleneck
    bn = conv3d_block(p3, base_filters * 8, dropout_rate=dropout_rate)

    # Decoder
    u3 = tf.keras.layers.UpSampling3D(2)(bn)           # 8→16
    u3 = tf.keras.layers.Concatenate()([u3, c3])
    c4 = conv3d_block(u3, base_filters * 4, dropout_rate=dropout_rate)

    u2 = tf.keras.layers.UpSampling3D(2)(c4)           # 16→32
    u2 = tf.keras.layers.Concatenate()([u2, c2])
    c5 = conv3d_block(u2, base_filters * 2)

    u1 = tf.keras.layers.UpSampling3D(2)(c5)           # 32→64
    u1 = tf.keras.layers.Concatenate()([u1, c1])
    c6 = conv3d_block(u1, base_filters)

    # Output: one sigmoid value per voxel
    outputs = tf.keras.layers.Conv3D(1, 1, activation="sigmoid")(c6)

    return tf.keras.Model(inputs, outputs, name="unet_3d_brats_peds")

def model2(*hparams):
    # reserve for Swin implementation
    pass

def conv3d_block(x, filters, dropout_rate=0.0):
    x = tf.keras.layers.Conv3D(filters, 3, padding="same", use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation("relu")(x)

    x = tf.keras.layers.Conv3D(filters, 3, padding="same", use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation("relu")(x)

    if dropout_rate > 0:
        x = tf.keras.layers.Dropout(dropout_rate)(x, training=True)
    return x