from segment_net.baseline import fcn_model
from segment_net.fcn_baseline import fcn_baseline


NETWORKS = {'fcn_model': fcn_model, 'fcn_baseline': fcn_baseline}


def get_network(name):
  assert name in NETWORKS, 'unimplemented %s' % name
  return NETWORKS[name]
