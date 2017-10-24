import os
import keras
import json
from segment_net.keras_fcn import fcn_model
from segment_net.network import KerasWrapper
from segment_net.flags import *

tf.flags.DEFINE_string(
  'output', None, 'output h5 and config directory')


def read_ckpt_weights():
  tf_model = KerasWrapper(
    name='fcn', path=FLAGS.save)
  return tf_model.variables


def assign_variables_to_keras(variables):

  model_ctr = fcn_model(name='fcn')
  eval_model = model_ctr()

  assert isinstance(eval_model, keras.engine.Model)

  sess = keras.backend.get_session()
  weights = eval_model.weights

  assign_ops = []

  for (a, v), b in zip(variables, weights):
    assert a.name == b.name
    op = b.assign(v)
    assign_ops.append(op)

  sess.run(assign_ops)
  return eval_model


def save_as_h5(model):
  if not os.path.exists(FLAGS.output):
    os.mkdir(FLAGS.output)
  assert isinstance(model, keras.engine.Model)
  config_path = os.path.join(FLAGS.output, 'config_fcn')
  weights_path = os.path.join(FLAGS.output, 'fcn')
  with open(config_path, 'w') as op:
    json.dump(model.to_json(), op)
  model.save_weights(weights_path)


def main(_):
  variables = read_ckpt_weights()
  model = assign_variables_to_keras(variables)
  save_as_h5(model)


if __name__ == '__main__':
    tf.app.run()
