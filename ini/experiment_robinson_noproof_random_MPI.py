import sys, os
sys.path.append(os.path.join(os.getcwd(), 'deps/awarelib'))
from awarelib.flop_tools import flop_simple_handle_experiment
from munch import Munch

base_config = Munch(actions_added=False,
                    actorbatch=512,
                    add_rnd_steps_to_remain=0,
                    can_replace_proof=True,
                    entcoeff=0.01,
                    evaldirs=["theorems/robinson/random/final2"],
                    evalprobcount=100,
                    evaltime=60,
                    failure_reward=0,
                    fast_features=False,
                    feature_file=None,
                    gamma=0.95,
                    illegal_reward=0,
                    known_proof_max_exploration=0,
                    latent_dim=None,
                    lr_schedule="constant",
                    max_exploration=60,
                    n_action_slots=22,
                    n_dim=100,
                    network_layers=[512,512],
                    optim_epochs=4,
                    optim_stepsize=0.00001,
                    outdir="results/robinson_noproof_random",
                    proof_allowed=True,
                    proof_dir="noproof",
                    quick_progress_threshold=None,
                    quick_progress_percentage=0.9,
                    same_file_count=1,
                    saved_model=None,
                    scheduler_starting_step=1,
                    scheduler_type="global",
                    steps_per_curriculum=10000,
                    supervised_reward=0,
                    terminate_on_illegal=False,
                    train_timesteps=[50000000],
                    train_dirs=["theorems/robinson/random/random2"],
                    use_previous_action=True,
                    value_gets_actions=False,
                    wd=0.0,
                    neptune=True,
                    tags=["random", "noproof", "final"])


params_grid = dict(proof_allowed=[True, False],
                   scheduler_type=["global", "local"],
                   train_dirs=[["theorems/robinson/random/tiny_one_op"],
#                               ["theorems/robinson/robinson_noproof/random2"],
                               ["theorems/robinson/random/tiny"]])


def spec():
  return flop_simple_handle_experiment(experiment_name = 'robinson_noproof_random',
                                     project_name = "deepmath/curriculum-tp",
                                     script='mpirun python train_ppo.py',
                                     python_path='.:deps/awarelib',
                                     paths_to_dump = '',
                                     exclude = [".git", ".gitignore", ".gitmodules", "log", "tree_methods", "results"],
                                     project_tag = "test",
                                     base_config=base_config,
                                     params_grid=params_grid,
                                     _script_name=globals()["script"])
