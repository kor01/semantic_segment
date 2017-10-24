import tensorflow as tf

# fcn flags
tf.flags.DEFINE_integer('encode_filters', 64, 'number pf filters')
tf.flags.DEFINE_integer('decode_filters', 64,
                        'number of filters in decoder phase')
tf.flags.DEFINE_integer('num_encoders', 4, 'number of encoding layer')
tf.flags.DEFINE_integer('scale', 2, 'scale in pyramid')

FLAGS = tf.flags.FLAGS
