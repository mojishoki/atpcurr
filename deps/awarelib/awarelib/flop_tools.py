from mrunner.experiment import Experiment
from awarelib.running_utils.namesgenerator import get_random_name
from awarelib.running_utils.spec_utils import get_git_head_info, get_combinations
import copy
import pathlib


def flop_simple_handle_experiment(experiment_name, base_config,
                                project_name, params_grid,
                                script, python_path,
                                paths_to_dump, exclude, project_tag, update_lambda=lambda d1, d2: d1.update(d2),
                                _script_name=None):

  random_tag = get_random_name()
  _script_name = None if _script_name is None else pathlib.Path(_script_name).stem
  tags = ["flop", project_tag, random_tag]
  if _script_name:
    tags.append(_script_name)
  params_configurations = get_combinations(params_grid)
  base_config['git_head'] = get_git_head_info()
  experiments = []
  for params_configuration in params_configurations:
    config = copy.deepcopy(base_config)
    update_lambda(config, params_configuration)
    experiments.append(Experiment(project=project_name, name=experiment_name, script=script,
                                  parameters=config, python_path=python_path,
                                  paths_to_dump=paths_to_dump, tags=tags,
                                  exclude=exclude))

  return experiments
