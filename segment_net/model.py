from segment_net.fcn_baseline import fcn_baseline
from segment_net.fcn import apply_fcn


NETWORKS = {'fcn_baseline': fcn_baseline, 'fcn': apply_fcn}


def get_network(name):
  assert name in NETWORKS, 'unimplemented %s' % name
  return NETWORKS[name]
