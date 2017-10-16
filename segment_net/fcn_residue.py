import tensorflow as tf
from utils.separable_conv2d import BilinearUpSampling2D

FLAGS = tf.flags.FLAGS
tf.flags.DEFINE_integer('filters', 32, 'number pf filters')


def encoder_block(img):

  with tf.variable_scope('encoder'):

    conv1 = tf.layers.conv2d(
      img, filters=FLAGS.filters, kernel_size=3,
      padding='SAME', activation=tf.nn.relu)
    pool1 = tf.layers.max_pooling2d(conv1, 2, 2)

    conv2 = tf.layers.conv2d(
      pool1, filters=FLAGS.filters,
      kernel_size=3, padding='SAME', activation=tf.nn.relu)
    pool2 = tf.layers.max_pooling2d(conv2, 2, 2)

    conv3 = tf.layers.conv2d(
      pool2, filters=FLAGS.filters,
      kernel_size=3, padding='SAME', activation=tf.nn.relu)
    pool3 = tf.layers.max_pooling2d(conv3, 2, 2)

    transpose1 = BilinearUpSampling2D((2, 2))(pool1)
    transpose2 = BilinearUpSampling2D((4, 4))(pool2)
    transpose3 = BilinearUpSampling2D((8, 8))(pool3)

    '''
    transpose1 = tf.layers.conv2d_transpose(
      pool1, FLAGS.filters, 2, 2, activation=tf.nn.relu)

    transpose2 = tf.layers.conv2d_transpose(
      pool2, FLAGS.filters, 4, 4, activation=tf.nn.relu)
    '''

  return [img, transpose1, transpose2, transpose3]


def decode_block(feature_maps):

  with tf.variable_scope('decoder'):
    features = tf.concat(feature_maps, axis=-1)

    features = tf.layers.conv2d(
      features, kernel_size=3, filters=128,
      padding='SAME', activation=tf.nn.relu)

    features = tf.layers.conv2d(
      features, kernel_size=3, filters=128,
      padding='SAME', activation=tf.nn.relu)

    logits = tf.layers.conv2d(
      features, filters=3, kernel_size=1, strides=1)
  return logits


def fcn_residue(images):
  encodes = encoder_block(images)
  logits = decode_block(encodes)
  return logits
