import os
import glob
import sys
import tensorflow as tf

from scipy import misc
import numpy as np

from tensorflow.contrib.keras.python import keras
from tensorflow.contrib.keras.python.keras import layers, models

from tensorflow import image

from utils import scoring_utils
from utils.separable_conv2d import SeparableConv2DKeras, BilinearUpSampling2D
from utils import data_iterator
from utils import plotting_tools
from utils import model_tools


def separable_conv2d_batchnorm(input_layer, filters, strides=1):
  output_layer = SeparableConv2DKeras(filters=filters, kernel_size=3, strides=strides,
                                      padding='same', activation='relu')(input_layer)
  output_layer = tf.layers.batch_normalization(output_layer, training=True)
  # output_layer = layers.BatchNormalization()(output_layer)
  return output_layer


def conv2d_batchnorm(input_layer, filters, kernel_size=3, strides=1):
  output_layer = layers.Conv2D(filters=filters, kernel_size=kernel_size, strides=strides,
                               padding='same', activation='relu')(input_layer)
  output_layer = tf.layers.batch_normalization(output_layer, training=True)
  # output_layer = layers.BatchNormalization()(output_layer)
  return output_layer


def bilinear_upsample(input_layer):
  output_layer = BilinearUpSampling2D((2, 2))(input_layer)
  return output_layer


def encoder_block(input_layer, filters, strides):
  output_layer = separable_conv2d_batchnorm(input_layer, filters, strides)
  return output_layer


def decoder_block(small_ip_layer, large_ip_layer, filters):
  upsampled_small_ip_layer = bilinear_upsample(small_ip_layer)
  concat_layer = layers.concatenate([upsampled_small_ip_layer, large_ip_layer])
  output_layer = separable_conv2d_batchnorm(concat_layer, filters)
  output_layer = separable_conv2d_batchnorm(output_layer, filters)
  return output_layer


def fcn_model(inputs, num_classes):

  encoder1 = encoder_block(inputs, filters=16, strides=2)
  encoder2 = encoder_block(encoder1, filters=32, strides=2)
  encoder3 = encoder_block(encoder2, filters=64, strides=2)

  conv_1x1 = conv2d_batchnorm(encoder3, filters=128, kernel_size=1, strides=1)
  decoder1 = decoder_block(small_ip_layer=conv_1x1, large_ip_layer=encoder2, filters=64)
  decoder2 = decoder_block(small_ip_layer=decoder1, large_ip_layer=encoder1, filters=32)
  decoder3 = decoder_block(small_ip_layer=decoder2, large_ip_layer=inputs, filters=16)
  return layers.Conv2D(num_classes, 1, padding='same')(decoder3)

