import tensorflow as tf


tf.flags.DEFINE_integer('encode_filters', 64, 'number pf filters')
tf.flags.DEFINE_integer('decode_filters', 64, 'number of filters in decoder phase')
tf.flags.DEFINE_integer('num_encoders', 4, 'number of encoding layer')
tf.flags.DEFINE_integer('scale', 2, 'scale in pyramid')

FLAGS = tf.flags.FLAGS


def encoder(img, scale):

  with tf.variable_scope('encoder'):

    def conv_and_pool(feature):
      conv = tf.layers.separable_conv2d(
        feature, FLAGS.filters, 3, 1,
        padding='SAME', activation=tf.nn.relu)
      pool = tf.layers.max_pooling2d(
        conv, scale, scale, padding='SAME')
      return pool

    pools = [img]

    for i in range(FLAGS.num_encoders):
      with tf.variable_scope('encoder-%d' % i):
        pools.append(conv_and_pool(pools[-1]))

    pools.reverse()
    return pools


def decoder(pools, scale, num_classes):
  filters = FLAGS.decode_filters

  last_pool = pools[0]
  for pool in pools[1:]:
    last_pool = tf.layers.conv2d_transpose(
      last_pool, filters, scale, scale, activation=tf.nn.relu)
    pool = tf.concat([pool, last_pool], axis=-1)
    last_pool = tf.layers.conv2d(
      pool, filters=filters, kernel_size=3,
      padding='SAME', activation=tf.nn.relu)

  logits = tf.layers.conv2d(
    last_pool, filters=num_classes, kernel_size=1)

  return logits


def apply_fcn(images, num_classes, is_train):

  with tf.variable_scope('fcn'):
    pools = encoder(images, FLAGS.scale)
    logits = decoder(pools, FLAGS.scale, num_classes)
  return logits
