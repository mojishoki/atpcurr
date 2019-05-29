import sys, os
sys.path.append(os.path.join(os.getcwd(), 'deps/awarelib'))
from awarelib.flop_tools import flop_simple_handle_experiment
from munch import Munch

base_config = Munch(actions_added=False,
                    actorbatch=512,
                    add_rnd_steps_to_remain=12,
                    can_replace_proof=False,
                    entcoeff=0.01,
                    evaldirs=["theorems/robinson/simple/final/"],
                    evalprobcount=100,
                    evaltime=60,
                    failure_reward=0,
                    fast_features=True,
                    feature_file=None,
                    gamma=0.99,
                    illegal_reward=0,
                    known_proof_max_exploration=0,
                    latent_dim=None,
                    lr_schedule="constant",
                    max_exploration=400,
                    n_action_slots=22,
                    n_dim=100,
                    network_layers=[512,512],
                    optim_epochs=4,
                    optim_stepsize=0.00001,
                    outdir="results/baseline_robinson_simple_suboptimal",
                    proof_allowed=True,
                    proof_dir="noproof",
                    quick_progress_threshold=None,
                    quick_progress_percentage=0.9,
                    same_file_count=1,
                    saved_model=None,
                    scheduler_starting_step=1,
                    scheduler_type="global",
                    steps_per_curriculum=200000000,
                    supervised_reward=0.1,
                    terminate_on_illegal=False,
                    train_timesteps=[20000000],
                    train_dirs=["theorems/robinson/robinson_noproof/simple2"],
                    use_previous_action=False,
                    value_gets_actions=False,
                    wd=0.0,
                    neptune=True,
                    tags=["simple", "supervised", "final"])


params_grid = dict(proof_dir=["theorems/proofs/fcoplib_20181109", "theorems/proofs/altproof1", "theorems/proofs/altproof2"])


def spec():
  return flop_simple_handle_experiment(experiment_name = 'robinson_simple_suboptimal',
                                     project_name = "deepmath/curriculum-tp",
                                     script='mpirun python train_ppo.py',
                                     python_path='.:deps/awarelib',
                                     paths_to_dump = '',
                                     exclude = [".git", ".gitignore", ".gitmodules", "log", "tree_methods", "results"],
                                     project_tag = "test",
                                     base_config=base_config,
                                     params_grid=params_grid,
                                     _script_name=globals()["script"])
