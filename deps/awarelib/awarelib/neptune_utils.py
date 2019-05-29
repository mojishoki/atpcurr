import argparse
import os
from deepsense import neptune
import socket
from munch import Munch
from path import Path

def is_neptune_online():
  # I wouldn't be suprised if this would depend on neptune version
  return 'NEPTUNE_ONLINE_CONTEXT' in os.environ

_ctx = None


def _ensure_compund_type(input):
  if type(input) is not str:
    return input

  try:
    input = eval(input, {})
    if type(input) is tuple or type(input) is list:
      return input
    else:
      return input
  except:
    return input

def get_configuration(print_diagnostics=False, dict_prefixes=[]):
  global _ctx
  if is_neptune_online():
    ctx = neptune.Context()
    exp_dir_path = os.getcwd()
    params = {k: _ensure_compund_type(ctx.params[k]) for k in ctx.params}
  else:
    # local run
    parser = argparse.ArgumentParser(description='Debug run.')
    parser.add_argument('--ex', type=str)
    parser.add_argument("--exp_dir_path", default='/tmp')
    commandline_args = parser.parse_args()
    if commandline_args.ex != None:
      script = commandline_args.ex
      vars = {'script': str(Path(commandline_args.ex).name)}
      exec(open(commandline_args.ex).read(), vars)
      spec_func = vars['spec']
      # take first experiment (params configuration)
      experiment = spec_func()[0]
      params = experiment.parameters
    else:
      params = {}
    # create offline context
    ctx = neptune.Context(offline_parameters=params)
    exp_dir_path = commandline_args.exp_dir_path
  _ctx = ctx
  ctx.properties['pwd'] = os.getcwd()
  ctx.properties['host'] = socket.gethostname()

  for dict_prefix in dict_prefixes:
    dict_params = {}
    l = len(dict_prefix)

    for k in list(params.keys()):
      if dict_prefix in k:
        dict_params[k[l:]] = params.pop(k)

    params[dict_prefix[:-1]] = dict_params

  if print_diagnostics:
    print("PYTHONPATH:{}".format(os.environ['PYTHONPATH']))
    print("cd {}".format(os.getcwd()))
    print(socket.gethostname())
    print("Params:{}".format(params))

  return Munch(params), exp_dir_path


def neptune_logger(m, v):
  global _ctx

  assert _ctx is not None, "Run first get_configuration"

  if _ctx.experiment_id is None:
    print("{}:{}".format(m, v))
  else:
    _ctx.channel_send(name=m, x=None, y=v)