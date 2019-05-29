import time

from stable_baselines.common.vec_env import DummyVecEnv
from ppo import PPO1
import eval

from fcop_env import ProofEnv, StageScheduler, Container
from ppo_model import FeedForwardPolicy
import params
import util
from awarelib.neptune_utils import get_configuration
from common_utils import openai_baselines_hack
from common_utils.general import get_log_dir
from baselines import logger
from common_utils.neptune import Neptune2OutputFormat, set_neptune_properties_for_experiment
import numpy as np

import os

# # load parameters
# args = params.getArgs()
# print("\n\n")
# for k in args.keys():
#     print(k, ": ", args[k])
# print("\n\n")

def main():

    args, _ = get_configuration(print_diagnostics=True)
    if args.use_previous_action:
        args.state_dim = 5
    else:
        args.state_dim = 4

    if not os.path.exists(args.outdir):
        os.makedirs(args.outdir)

    class MyMlpPolicy(FeedForwardPolicy):
        def __init__(self, sess, ob_space, ac_space, n_env, n_steps, n_batch, reuse=False, **_kwargs):
            super().__init__(sess,
                             ob_space,
                             ac_space,
                             n_env,
                             n_steps,
                             n_batch,
                             reuse=reuse,
                             feature_extraction="mlp",
                             layers=args.network_layers,
                             args=args, **_kwargs)

    t0 = time.time()

    stage_scheduler = StageScheduler(args)
    container = Container(args, stage_scheduler=stage_scheduler)


    # The algorithms require a vectorized environment to run
    env = DummyVecEnv([lambda: ProofEnv(args, container, stage_scheduler)])

    if args.saved_model == None:
        env.envs[0].set_directory(args.train_dirs[0])
        model = PPO1(MyMlpPolicy, env,
                     verbose=2,
                     timesteps_per_actorbatch=args.actorbatch,
                     schedule=args.lr_schedule,
                     optim_stepsize=args.optim_stepsize,
                     entcoeff=args.entcoeff,
                     wd=args.wd,
                     gamma=args.gamma,
                     tensorboard_log="log/ppo1")
    else:
        print("Loading model from {}".format(args.saved_model))
        model = PPO1.load(args.saved_model)
        model.set_env(env)

    openai_baselines_hack.monkey_patch()

    from mpi4py import MPI as mpi
    comm = mpi.COMM_WORLD
    rank = comm.Get_rank()
    all = comm.Get_size()
    print("My rank is {} out of {}".format(rank, all))
    #
    # def mpi_sum(value):
    #     global_sum = np.zeros(1, dtype='float64')
    #     local_sum = np.sum(np.array(value)).astype('float64')
    #     mpi.COMM_WORLD.Reduce(local_sum, global_sum, op=mpi.SUM)
    #     return global_sum[0]

    logdir = get_log_dir('atpcurr')
    print("Logdir: {}".format(logdir))

    custom_output_formats = []

    if args.neptune and rank == 0:
        from deepsense import neptune
        neptune_ctx = neptune.Context()

        for tag in args.tags:
            neptune_ctx.tags.append(tag.lower())

        set_neptune_properties_for_experiment(neptune_ctx, args, log_dir=logdir)
        custom_output_formats.append(Neptune2OutputFormat(neptune_ctx))

    logger.configure(
        dir=logdir,
        format_strs=['stdout', 'tensorboard'],
        custom_output_formats=custom_output_formats
    )

    container.model = model
    counter = 0

    modelfiles = []
    for train_timestep, train_dir in zip(args.train_timesteps, args.train_dirs):
        env.envs[0].set_directory(train_dir)
        model.learn(total_timesteps=train_timestep, callback=container.callback)
        print("Training on {} finished in {}".format(train_dir, util.format_time(time.time() - t0)))
        util.print_problemdict(env.envs[0].problemdict)
        modelfile = "{}/ppo1_fcop_train_{}".format(args.outdir, counter)
        modelfiles.append(modelfile)
        if rank == 0:
            model.save(modelfile)
            # logger.logkv("finished_train_problems", counter)
        counter += 1

    print("We have finished training, rank {}".format(rank))
    # here we wait for everyone
    comm.Barrier()
    print("We have started evaluation, rank {}".format(rank))

    # evaluation without training
    if (args.saved_model is not None) and (len(args.train_dirs) == 0): # no training, just evaluation
        modelfiles = [args.saved_model]

    for evaldir in args.evaldirs:
        for model_index, modelfile in enumerate(modelfiles):

            # args.evaltype = "det"
            # args.evalcount = 1
            # eval.eval_mpi(args, evaldir, modelfile, model_index)

            # # here we wait for everyone
            # comm.Barrier()

            args.evaltype = "prob"
            args.evalcount = args.evalprobcount
            eval.eval_mpi(args, evaldir, modelfile, model_index)

            # here we wait for everyone
            comm.Barrier()

if __name__ == '__main__':
    main()
