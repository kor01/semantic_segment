from os.path import join
from keras import layers
from keras.engine import Model
from segment_net.flags import *
from segment_net.fcn_flags import *


def conv_and_pool(name):

  conv_layer = layers.SeparableConv2D(
      filters=FLAGS.encode_filters,
      kernel_size=3, padding='SAME',
      activation='relu', name=join(name, 'separable_conv2d'))
  pool_layer = layers.MaxPool2D(
      pool_size=FLAGS.scale,
      strides=FLAGS.scale, padding='SAME',
      name=join(name, 'max_pool'))

  def conv_and_pool_fn(feature_map):
    conv = conv_layer(feature_map)
    pool = pool_layer(conv)
    return pool
  return conv_and_pool_fn


def encoder(name):

  fns = [conv_and_pool(
    join(name, 'encoder-%d' % i))
    for i in range(FLAGS.num_encoders)]

  def encoder_fn(img):
    pools = [img]
    for fn in fns:
      pools.append(fn(pools[-1]))
    pools.reverse()
    return pools

  return encoder_fn


def transpose_concat_conv(name):

  filters = FLAGS.decode_filters
  transpose_layer = layers.Conv2DTranspose(
        filters=filters, kernel_size=2,
        strides=2, activation='relu',
        name=join(name, 'conv2d_transpose'))
  concat_layer = layers.Concatenate(axis=-1)
  conv_layer = layers.Conv2D(
        filters=filters, kernel_size=3,
        padding='SAME', activation='relu',
        name=join(name, 'conv2d'))

  def tcc_fn(last_pool, pool):
    last_pool = transpose_layer(last_pool)
    pool = concat_layer([pool, last_pool])
    last_pool = conv_layer(pool)
    return last_pool

  return tcc_fn


def decoder(name):

  tccs = [transpose_concat_conv(
    join(name, 'decoder-%d' % i))
    for i in range(FLAGS.num_encoders)]

  one_by_one = layers.Conv2D(
    filters=FLAGS.classes, kernel_size=1,
    activation='softmax', name=join(name, 'one_by_one'))

  def decoder_fn(pools):
    assert len(pools) == FLAGS.num_encoders + 1
    last_pool = pools[0]
    for pool, tcc in zip(pools[1:], tccs):
      last_pool = tcc(last_pool, pool)
    prob = one_by_one(last_pool)

    return prob
  return decoder_fn


def fcn_model(name):

  encoder_fn = encoder(name)
  decoder_fn = decoder(name)

  def create_fn(inputs=None):
    if inputs is None:
      inputs = layers.Input(shape=(FLAGS.height, FLAGS.width, 3))
    else:
      inputs = inputs + (3,)
      inputs = layers.Input(shape=inputs)
    prob = decoder_fn(encoder_fn(inputs))
    model = Model(inputs=inputs, outputs=prob)
    return model

  return create_fn
