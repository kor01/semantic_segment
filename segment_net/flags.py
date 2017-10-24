import tensorflow as tf

# train flags

tf.flags.DEFINE_string('log', None, 'log file, None to stderr')
tf.flags.DEFINE_string('save', None, 'model ckpt path')
tf.flags.DEFINE_integer('epoch', 500, 'number of epoch')
tf.flags.DEFINE_integer('evaluate_step', 4, 'epoch steps for evaluation')
tf.flags.DEFINE_float('lr', 1e-3, 'learning rate')
tf.flags.DEFINE_string('name', 'fcn_model', 'name of network')

tf.flags.DEFINE_string('train', None, 'train images and masks directory')
tf.flags.DEFINE_string('valid', None, 'validate images and masks directory')
tf.flags.DEFINE_string('evaluate', None, 'sample evaluation dataset')


# network definition flags

tf.flags.DEFINE_integer('width', 256, 'default image width')
tf.flags.DEFINE_integer('height', 256, 'default image height')
tf.flags.DEFINE_integer('classes', 3, 'number of semantic classes')


# dataset flags

tf.flags.DEFINE_boolean('normalize', True, 'normalize image')
tf.flags.DEFINE_integer('threads', 4, 'number of preprocess threads')
tf.flags.DEFINE_integer('crop_size', 64, 'crop size')
tf.flags.DEFINE_integer('batch_size', 128, 'number pf filters')
tf.flags.DEFINE_integer('cache_batch', 128, 'batch to cache in memory')


FLAGS = tf.flags.FLAGS
