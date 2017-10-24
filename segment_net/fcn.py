from segment_net.fcn_flags import *


def encoder(img, scale):

  def conv_and_pool(feature):
    conv = tf.layers.separable_conv2d(
      feature, filters=FLAGS.encode_filters,
      kernel_size=3, strides=1,
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
  for i, pool in enumerate(pools[1:]):
    with tf.variable_scope('decoder-%d' % i):
      last_pool = tf.layers.conv2d_transpose(
        last_pool, filters, scale, scale, activation=tf.nn.relu)
      pool = tf.concat([pool, last_pool], axis=-1)
      last_pool = tf.layers.conv2d(
        pool, filters=filters, kernel_size=3,
        padding='SAME', activation=tf.nn.relu)

  logits = tf.layers.conv2d(
    last_pool, filters=num_classes,
    kernel_size=1, name='one_by_one')

  return logits


def apply_fcn(images, num_classes, is_train):
  pools = encoder(images, FLAGS.scale)
  logits = decoder(pools, FLAGS.scale, num_classes)
  return logits
