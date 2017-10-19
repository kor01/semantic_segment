import tensorflow as tf


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

    transpose1 = tf.layers.conv2d_transpose(
      pool1, FLAGS.filters, 2, 2, activation=tf.nn.relu)

    transpose2 = tf.layers.conv2d_transpose(
      pool2, FLAGS.filters, 4, 4, activation=tf.nn.relu)

  return [conv1, transpose1, transpose2]


def decode_block(feature_maps, num_classes):

  with tf.variable_scope('decoder'):
    features = tf.concat(feature_maps, axis=-1)

    features = tf.layers.conv2d(
      features, kernel_size=3, filters=64,
      padding='SAME', activation=tf.nn.relu)

    features = tf.layers.conv2d(
      features, kernel_size=3, filters=64,
      padding='SAME', activation=tf.nn.relu)

    logits = tf.layers.conv2d(
      features, filters=num_classes,
      kernel_size=1, strides=1)
  return logits


def fcn_baseline(images, num_classes, is_train):
  with tf.variable_scope('fc_baseline'):
    encodes = encoder_block(images)
    logits = decode_block(encodes, num_classes)
  return logits
