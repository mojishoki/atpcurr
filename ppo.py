from collections import deque
import time

import tensorflow as tf
import numpy as np
from mpi4py import MPI

from stable_baselines.common import Dataset, explained_variance, fmt_row, zipsame, ActorCriticRLModel, SetVerbosity, \
        TensorboardWriter
# from stable_baselines import logger
from baselines import logger
import stable_baselines.common.tf_util as tf_util
from stable_baselines.common.policies import LstmPolicy, ActorCriticPolicy
from stable_baselines.common.mpi_adam import MpiAdam
from stable_baselines.common.mpi_moments import mpi_moments
from stable_baselines.trpo_mpi.utils import traj_segment_generator, flatten_lists
from stable_baselines.a2c.utils import total_episode_reward_logger

import copy


def add_vtarg_and_adv(seg, gamma, lam):
    """
    Compute target value using TD(lambda) estimator, and advantage with GAE(lambda)
    :param seg: (dict) the current segment of the trajectory (see traj_segment_generator return for more information)
    :param gamma: (float) Discount factor
    :param lam: (float) GAE factor
    """
    # last element is only used for last vtarg, but we already zeroed it if last new = 1
    new = seg["dones"]
    vpred = np.append(seg["vpred"], seg["nextvpred"])
    rew_len = len(seg["rew"])
    seg["adv"] = gaelam = np.empty(rew_len, 'float32')
    rew = seg["rew"]
    lastgaelam = 0
    for step in reversed(range(rew_len)):
        nonterminal = 1 - new[step]
        delta = rew[step] + gamma * vpred[step + 1] * nonterminal - vpred[step]
        gaelam[step] = lastgaelam = delta + gamma * lam * nonterminal * lastgaelam
    seg["tdlamret"] = seg["adv"] + seg["vpred"]


class PPO1(ActorCriticRLModel):
    """
    Proximal Policy Optimization algorithm (MPI version).
    Paper: https://arxiv.org/abs/1707.06347
    :param env: (Gym environment or str) The environment to learn from (if registered in Gym, can be str)
    :param policy: (ActorCriticPolicy or str) The policy model to use (MlpPolicy, CnnPolicy, CnnLstmPolicy, ...)
    :param timesteps_per_actorbatch: (int) timesteps per actor per update
    :param clip_param: (float) clipping parameter epsilon
    :param entcoeff: (float) the entropy loss weight
    :param optim_epochs: (float) the optimizer's number of epochs
    :param optim_stepsize: (float) the optimizer's stepsize
    :param optim_batchsize: (int) the optimizer's the batch size
    :param gamma: (float) discount factor
    :param lam: (float) advantage estimation
    :param adam_epsilon: (float) the epsilon value for the adam optimizer
    :param schedule: (str) The type of scheduler for the learning rate update ('linear', 'constant',
        'double_linear_con', 'middle_drop' or 'double_middle_drop')
    :param verbose: (int) the verbosity level: 0 none, 1 training information, 2 tensorflow debug
    :param tensorboard_log: (str) the log location for tensorboard (if None, no logging)
    :param _init_setup_model: (bool) Whether or not to build the network at the creation of the instance
    :param wd: (float) weight decay coefficient
    """

    def __init__(self, policy, env, gamma=0.99, timesteps_per_actorbatch=256, clip_param=0.2, entcoeff=0.01,
                 optim_epochs=4, optim_stepsize=1e-3, optim_batchsize=64, lam=0.95, adam_epsilon=1e-5,
                 schedule='linear', verbose=0, tensorboard_log=None, _init_setup_model=True, wd=0.0):
        
        super().__init__(policy=policy, env=env, verbose=verbose, requires_vec_env=False,
                         _init_setup_model=_init_setup_model)
            
        self.gamma = gamma
        self.timesteps_per_actorbatch = timesteps_per_actorbatch
        self.clip_param = clip_param
        self.entcoeff = entcoeff
        self.optim_epochs = optim_epochs
        self.optim_stepsize = optim_stepsize
        self.optim_batchsize = optim_batchsize
        self.lam = lam
        self.adam_epsilon = adam_epsilon
        self.schedule = schedule
        self.tensorboard_log = tensorboard_log
        self.wd = wd
            
        self.graph = None
        self.sess = None
        self.policy_pi = None
        self.loss_names = None
        self.lossandgrad = None
        self.adam = None
        self.assign_old_eq_new = None
        self.compute_losses = None
        self.params = None
        self.step = None
        self.proba_step = None
        self.initial_state = None
        self.summary = None
        self.episode_reward = None
            
        if _init_setup_model:
            self.setup_model()
                
    def setup_model(self):
        with SetVerbosity(self.verbose):
            
            self.graph = tf.Graph()
            with self.graph.as_default():
                self.sess = tf_util.single_threaded_session(graph=self.graph)
                
                # Construct network for new policy
                self.policy_pi = self.policy(self.sess, self.observation_space, self.action_space, self.n_envs, 1,
                                             None, reuse=False)
                            
                # Network for old policy
                with tf.variable_scope("oldpi", reuse=False):
                    old_pi = self.policy(self.sess, self.observation_space, self.action_space, self.n_envs, 1,
                                         None, reuse=False)
                                
                with tf.variable_scope("loss", reuse=False):
                    # Target advantage function (if applicable)
                    atarg = tf.placeholder(dtype=tf.float32, shape=[None])
                    
                    # Empirical return
                    ret = tf.placeholder(dtype=tf.float32, shape=[None])
                    
                    # learning rate multiplier, updated with schedule
                    lrmult = tf.placeholder(name='lrmult', dtype=tf.float32, shape=[])
                    
                    # Annealed cliping parameter epislon
                    clip_param = self.clip_param * lrmult
                    
                    obs_ph = self.policy_pi.obs_ph
                    action_ph = self.policy_pi.pdtype.sample_placeholder([None])
                    
                    kloldnew = old_pi.proba_distribution.kl(self.policy_pi.proba_distribution)
                    ent = self.policy_pi.proba_distribution.entropy()
                    meankl = tf.reduce_mean(kloldnew)
                    meanent = tf.reduce_mean(ent)
                    pol_entpen = (-self.entcoeff) * meanent
                                    
                    # pnew / pold
                    ratio = tf.exp(self.policy_pi.proba_distribution.logp(action_ph) -
                                   old_pi.proba_distribution.logp(action_ph))
                                    
                    # surrogate from conservative policy iteration
                    surr1 = ratio * atarg
                    surr2 = tf.clip_by_value(ratio, 1.0 - clip_param, 1.0 + clip_param) * atarg
                                    
                    # PPO's pessimistic surrogate (L^CLIP)
                    pol_surr = - tf.reduce_mean(tf.minimum(surr1, surr2))
                    vf_loss = tf.reduce_mean(tf.square(self.policy_pi.value_fn[:, 0] - ret))

                    #  weight decay
                    vars   = tf_util.get_trainable_vars("model") # tf.trainable_variables()
                    wd_loss = tf.add_n([ tf.nn.l2_loss(v) for v in vars if 'bias' not in v.name ]) * self.wd
                    
                    total_loss = pol_surr + pol_entpen + vf_loss + wd_loss
                    losses = [pol_surr, pol_entpen, vf_loss, meankl, meanent, wd_loss]
                    self.loss_names = ["pol_surr", "pol_entpen", "vf_loss", "kl", "ent", "wd"]
                                    
                    tf.summary.scalar('entropy_loss', pol_entpen)
                    tf.summary.scalar('policy_gradient_loss', pol_surr)
                    tf.summary.scalar('value_function_loss', vf_loss)
                    tf.summary.scalar('approximate_kullback-leiber', meankl)
                    tf.summary.scalar('clip_factor', clip_param)
                    tf.summary.scalar('loss', total_loss)
                                    
                    self.params = tf_util.get_trainable_vars("model")
                    
                    self.assign_old_eq_new = tf_util.function(
                        [], [], updates=[tf.assign(oldv, newv) for (oldv, newv) in
                                         zipsame(tf_util.get_globals_vars("oldpi"), tf_util.get_globals_vars("model"))])
                                    
                with tf.variable_scope("Adam_mpi", reuse=False):
                    self.adam = MpiAdam(self.params, epsilon=self.adam_epsilon, sess=self.sess)
                                        
                with tf.variable_scope("input_info", reuse=False):
                    tf.summary.scalar('discounted_rewards', tf.reduce_mean(ret))
                    tf.summary.histogram('discounted_rewards', ret)
                    tf.summary.scalar('learning_rate', tf.reduce_mean(self.optim_stepsize))
                    tf.summary.histogram('learning_rate', self.optim_stepsize)
                    tf.summary.scalar('advantage', tf.reduce_mean(atarg))
                    tf.summary.histogram('advantage', atarg)
                    tf.summary.scalar('clip_range', tf.reduce_mean(self.clip_param))
                    tf.summary.histogram('clip_range', self.clip_param)
                    if len(self.observation_space.shape) == 3:
                        tf.summary.image('observation', obs_ph)
                    else:
                        tf.summary.histogram('observation', obs_ph)
                                                
                self.step = self.policy_pi.step
                self.proba_step = self.policy_pi.proba_step
                self.initial_state = self.policy_pi.initial_state
                
                tf_util.initialize(sess=self.sess)
                                                
                self.summary = tf.summary.merge_all()
                
                self.lossandgrad = tf_util.function([obs_ph, old_pi.obs_ph, action_ph, atarg, ret, lrmult],
                                                    [self.summary, tf_util.flatgrad(total_loss, self.params)] + losses)
                self.compute_losses = tf_util.function([obs_ph, old_pi.obs_ph, action_ph, atarg, ret, lrmult],
                                                       losses)
                                                
    def learn(self, total_timesteps, callback=None, seed=None, log_interval=100, tb_log_name="PPO1"):
        with SetVerbosity(self.verbose), TensorboardWriter(self.graph, self.tensorboard_log, tb_log_name) as writer:
            self._setup_learn(seed)
                                
            assert issubclass(self.policy, ActorCriticPolicy), "Error: the input policy for the PPO1 model must be " \
                "an instance of common.policies.ActorCriticPolicy."
                            
            with self.sess.as_default():
                self.adam.sync()
                
                # Prepare for rollouts
                seg_gen = traj_segment_generator(self.policy_pi, self.env, self.timesteps_per_actorbatch)
                # seg_gen = filtered_traj_segment_generator(self.policy_pi, self.env, self.timesteps_per_actorbatch, imbalance_limit = self.timesteps_per_actorbatch // 100, waste_limit=self.timesteps_per_actorbatch*10)
                # seg_gen = balanced_traj_segment_generator(self.policy_pi, self.env, self.timesteps_per_actorbatch, waste_limit=self.timesteps_per_actorbatch*10)
                
                episodes_so_far = 0
                timesteps_so_far = 0
                iters_so_far = 0
                t_start = time.time()
                
                # rolling buffer for episode lengths
                lenbuffer = deque(maxlen=100)
                # rolling buffer for episode rewards
                rewbuffer = deque(maxlen=100)
                
                self.episode_reward = np.zeros((self.n_envs,))
                
                while True:
                    if callback is not None:
                        # Only stop training if return value is False, not when it is None. This is for backwards
                        # compatibility with callbacks that have no return statement.
                        if callback(locals(), globals()) == False:
                            break
                    if total_timesteps and timesteps_so_far >= total_timesteps:
                        break
                    
                    if self.schedule == 'constant':
                        cur_lrmult = 1.0
                    elif self.schedule == 'linear':
                        cur_lrmult = max(1.0 - float(timesteps_so_far) / total_timesteps, 0)
                    else:
                        raise NotImplementedError
                    
                    logger.log("********** Iteration %i ************" % iters_so_far)
                    # logger.record_tabular("update_no", iters_so_far)
                    logger.logkv("update_no", iters_so_far)

                    
                    seg = seg_gen.__next__()
                    add_vtarg_and_adv(seg, self.gamma, self.lam)
                    # seg = balanced_sample(seg_gen, self.timesteps_per_actorbatch, self.gamma, self.lam, waste_limit=self.timesteps_per_actorbatch*10)
                    
                    # ob, ac, atarg, ret, td1ret = map(np.concatenate, (obs, acs, atargs, rets, td1rets))
                    obs_ph, action_ph, atarg, tdlamret = seg["ob"], seg["ac"], seg["adv"], seg["tdlamret"]
                    
                    # true_rew is the reward without discount
                    if writer is not None:
                        self.episode_reward = total_episode_reward_logger(self.episode_reward,
                                                                          seg["true_rew"].reshape((self.n_envs, -1)),
                                                                          seg["dones"].reshape((self.n_envs, -1)),
                                                                          writer, timesteps_so_far)
                        
                    # predicted value function before udpate
                    vpredbefore = seg["vpred"]
                    
                    # standardized advantage function estimate
                    atarg = (atarg - atarg.mean()) / atarg.std()
                    dataset = Dataset(dict(ob=obs_ph, ac=action_ph, atarg=atarg, vtarg=tdlamret),
                                      shuffle=not issubclass(self.policy, LstmPolicy))
                    optim_batchsize = self.optim_batchsize or obs_ph.shape[0]
                    
                    # set old parameter values to new parameter values
                    self.assign_old_eq_new(sess=self.sess)
                    logger.log("Optimizing...")
                    logger.log(fmt_row(13, self.loss_names))
                    
                    # Here we do a bunch of optimization epochs over the data
                    for k in range(self.optim_epochs):
                        # list of tuples, each of which gives the loss for a minibatch
                        losses = []
                        for i, batch in enumerate(dataset.iterate_once(optim_batchsize)):
                            steps = (timesteps_so_far +
                                     k * optim_batchsize +
                                     int(i * (optim_batchsize / len(dataset.data_map))))
                            if writer is not None:
                                # run loss backprop with summary, but once every 10 runs save the metadata
                                # (memory, compute time, ...)
                                if (1 + k) % 10 == 0:
                                    run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                                    run_metadata = tf.RunMetadata()
                                    summary, grad, *newlosses = self.lossandgrad(batch["ob"], batch["ob"], batch["ac"],
                                                                                 batch["atarg"], batch["vtarg"],
                                                                                 cur_lrmult, sess=self.sess,
                                                                                 options=run_options,
                                                                                 run_metadata=run_metadata)
                                    writer.add_run_metadata(run_metadata, 'step%d' % steps)
                                else:
                                    summary, grad, *newlosses = self.lossandgrad(batch["ob"], batch["ob"], batch["ac"],
                                                                                 batch["atarg"], batch["vtarg"],
                                                                                 cur_lrmult, sess=self.sess)
                                writer.add_summary(summary, steps)
                            else:
                                _, grad, *newlosses = self.lossandgrad(batch["ob"], batch["ob"], batch["ac"],
                                                                       batch["atarg"], batch["vtarg"], cur_lrmult,
                                                                       sess=self.sess)
                                
                            self.adam.update(grad, self.optim_stepsize * cur_lrmult)
                            losses.append(newlosses)
                        logger.log(fmt_row(13, np.mean(losses, axis=0)))
                        
                    logger.log("Evaluating losses...")
                    losses = []
                    for batch in dataset.iterate_once(optim_batchsize):
                        newlosses = self.compute_losses(batch["ob"], batch["ob"], batch["ac"], batch["atarg"],
                                                        batch["vtarg"], cur_lrmult, sess=self.sess)
                        losses.append(newlosses)
                    mean_losses, _, _ = mpi_moments(losses, axis=0)
                    logger.log(fmt_row(13, mean_losses))
                    for (loss_val, name) in zipsame(mean_losses, self.loss_names):
                        logger.record_tabular("loss_" + name, loss_val)
                    logger.record_tabular("ev_tdlam_before", explained_variance(vpredbefore, tdlamret))
                    
                    # local values
                    lrlocal = (seg["ep_lens"], seg["ep_rets"])
                    
                    # list of tuples
                    listoflrpairs = MPI.COMM_WORLD.allgather(lrlocal)
                    lens, rews = map(flatten_lists, zip(*listoflrpairs))
                    lenbuffer.extend(lens)
                    rewbuffer.extend(rews)
                    logger.record_tabular("EpLenMean", np.mean(lenbuffer))
                    logger.record_tabular("EpRewMean", np.mean(rewbuffer))
                    logger.record_tabular("EpThisIter", len(lens))
                    episodes_so_far += len(lens)
                    timesteps_so_far += MPI.COMM_WORLD.allreduce(seg["total_timestep"])
                    iters_so_far += 1
                    logger.record_tabular("EpisodesSoFar", episodes_so_far)
                    logger.record_tabular("TimestepsSoFar", timesteps_so_far)
                    logger.record_tabular("TimeElapsed", time.time() - t_start)
                    if self.verbose >= 1 and MPI.COMM_WORLD.Get_rank() == 0:
                        logger.dump_tabular()
                        
        return self
    
    def save(self, save_path):
        data = {
            "gamma": self.gamma,
            "timesteps_per_actorbatch": self.timesteps_per_actorbatch,
            "clip_param": self.clip_param,
            "entcoeff": self.entcoeff,
            "optim_epochs": self.optim_epochs,
            "optim_stepsize": self.optim_stepsize,
            "optim_batchsize": self.optim_batchsize,
            "lam": self.lam,
            "adam_epsilon": self.adam_epsilon,
            "schedule": self.schedule,
            "verbose": self.verbose,
            "policy": self.policy,
            "observation_space": self.observation_space,
            "action_space": self.action_space,
            "n_envs": self.n_envs,
            "_vectorize_action": self._vectorize_action
        }
        
        params = self.sess.run(self.params)
        
        self._save_to_file(save_path, data=data, params=params)



# Returns states where the imbalance between positive and negative rewards is at most imbalance_limit
# if we have discarded more than waste_limit, then we stop caring about imbalance
def filtered_traj_segment_generator(policy, env, horizon, imbalance_limit=100, waste_limit=10000):

    observation = env.reset()
    action = env.action_space.sample()  # not used, just so we have the datatype
    
    # real rollout generator
    seg_gen = traj_segment_generator(policy, env, horizon)

    while True:
        # Initialize history arrays
        observations = np.array([observation for _ in range(horizon)])
        true_rews = np.zeros(horizon, 'float32')
        rews = np.zeros(horizon, 'float32')
        vpreds = np.zeros(horizon, 'float32')
        dones = np.zeros(horizon, 'int32')
        actions = np.array([action for _ in range(horizon)])
        prev_actions = actions.copy()

        ep_rets = []
        ep_lens = []
        ep_true_rets = []
        total_timesteps = 0
        nextvpreds = 0
    
        positive = 0
        negative = 0
        counter = 0
        wasted = 0
        
        while counter < horizon:
            result = next(seg_gen)
            total_timesteps += result["total_timestep"]
            nextvpreds = result["nextvpred"]
            ep_rets += result["ep_rets"]
            ep_lens += result["ep_lens"]
            ep_true_rets + result["ep_true_rets"]
            
            for i in range(horizon):
                isPositive = result["rew"][i] > 0
                if (wasted >= waste_limit) or\
                        (isPositive and positive - negative <= imbalance_limit) or \
                        (not isPositive and negative - positive <= imbalance_limit):
                    if isPositive:
                        positive +=1
                    else:
                        negative +=1
                    observations[counter] = result["ob"][i]
                    true_rews[counter] = result["true_rew"][i]
                    rews[counter] = result["rew"][i]
                    vpreds[counter] = result["vpred"][i]
                    dones[counter] = result["dones"][i]
                    actions[counter] = result["ac"][i]
                    prev_actions[counter] = result["prevac"][i]
                    counter +=1
                    if counter == horizon:
                        break
                else:
                    wasted += 1
        print("AAActor episode: {} positive, {} negative, {} wasted".format(positive, negative, wasted))
        logger.logkv("actor_positive", positive)
        logger.logkv("actor_negative", negative)
        logger.logkv("actor_", wasted)
        
        yield {"ob": observations, "rew": rews, "dones": dones, "true_rew": true_rews, "vpred": vpreds,
               "ac": actions, "prevac": prev_actions, "nextvpred": nextvpreds, "ep_rets": ep_rets,
               "ep_lens": ep_lens, "ep_true_rets": ep_true_rets, "total_timestep": total_timesteps}


# Returns totally balanced (pos-neg) dataset using oversampling
def balanced_traj_segment_generator(policy, env, horizon, waste_limit=10000):

    assert horizon % 2 == 0, "Horizon {} should be divisible by two".format(horizon)
    half_size = horizon // 2
    
    observation = env.reset()
    action = env.action_space.sample()  # not used, just so we have the datatype
    
    # real rollout generator
    seg_gen = traj_segment_generator(policy, env, horizon)

    while True:
        # Initialize history arrays
        pos_observations = np.array([observation for _ in range(horizon)])
        pos_true_rews = np.zeros(horizon, 'float32')
        pos_rews = np.zeros(horizon, 'float32')
        pos_vpreds = np.zeros(horizon, 'float32')
        pos_dones = np.zeros(horizon, 'int32')
        pos_actions = np.array([action for _ in range(horizon)])
        pos_prev_actions = pos_actions.copy()
        neg_observations = np.array([observation for _ in range(horizon)])
        neg_true_rews = np.zeros(horizon, 'float32')
        neg_rews = np.zeros(horizon, 'float32')
        neg_vpreds = np.zeros(horizon, 'float32')
        neg_dones = np.zeros(horizon, 'int32')
        neg_actions = np.array([action for _ in range(horizon)])
        neg_prev_actions = neg_actions.copy()

        ep_rets = []
        ep_lens = []
        ep_true_rets = []
        total_timesteps = 0
        nextvpreds = 0
    
        positive = 0
        negative = 0
        wasted = 0

        true_pos = 0
        true_neg = 0
        
        while positive + negative < horizon:
            result = next(seg_gen)
            total_timesteps += result["total_timestep"]
            nextvpreds = result["nextvpred"]
            ep_rets += result["ep_rets"]
            ep_lens += result["ep_lens"]
            ep_true_rets + result["ep_true_rets"]
            
            for i in range(horizon):
                # assert wasted < waste_limit, "Too much wasted data"
                
                isPositive = result["rew"][i] > 0
                if isPositive:
                    if positive < half_size: # positive sample and there is still space for it
                        pos_observations[positive] = result["ob"][i]
                        pos_true_rews[positive] = result["true_rew"][i]
                        pos_rews[positive] = result["rew"][i]
                        pos_vpreds[positive] = result["vpred"][i]
                        pos_dones[positive] = result["dones"][i]
                        pos_actions[positive] = result["ac"][i]
                        pos_prev_actions[positive] = result["prevac"][i]
                        true_pos +=1
                        positive +=1
                    elif negative > 0: # no more place for positive, and we already have some negative to sample from
                        copy_index = np.random.choice(negative)
                        neg_observations[negative] = neg_observations[copy_index]
                        neg_true_rews[negative] = neg_true_rews[copy_index]
                        neg_rews[negative] = neg_rews[copy_index]
                        neg_vpreds[negative] = neg_vpreds[copy_index]
                        neg_dones[negative] = neg_dones[copy_index]
                        neg_actions[negative] = neg_actions[copy_index]
                        neg_prev_actions[negative] = neg_prev_actions[copy_index]
                        if neg_rews[copy_index] > 0:
                            true_pos += 1
                        else:
                            true_neg += 1
                        negative +=1
                        wasted +=1
                    elif wasted > waste_limit: # no more tolerance, so we just pretend the sample was negative
                        neg_observations[negative] = result["ob"][i]
                        neg_true_rews[negative] = result["true_rew"][i]
                        neg_rews[negative] = result["rew"][i]
                        neg_vpreds[negative] = result["vpred"][i]
                        neg_dones[negative] = result["dones"][i]
                        neg_actions[negative] = result["ac"][i]
                        neg_prev_actions[negative] = result["prevac"][i]
                        true_pos +=1
                        negative +=1                        
                    else:
                        wasted +=1
                else:
                    if negative < half_size: # negative sample and there is still space for it
                        neg_observations[negative] = result["ob"][i]
                        neg_true_rews[negative] = result["true_rew"][i]
                        neg_rews[negative] = result["rew"][i]
                        neg_vpreds[negative] = result["vpred"][i]
                        neg_dones[negative] = result["dones"][i]
                        neg_actions[negative] = result["ac"][i]
                        neg_prev_actions[negative] = result["prevac"][i]
                        true_neg +=1
                        negative +=1
                    elif positive > 0: # no more place for negative, and we already have some positive to sample from
                        copy_index = np.random.choice(positive)
                        pos_observations[negative] = pos_observations[copy_index]
                        pos_true_rews[negative] = pos_true_rews[copy_index]
                        pos_rews[negative] = pos_rews[copy_index]
                        pos_vpreds[negative] = pos_vpreds[copy_index]
                        pos_dones[negative] = pos_dones[copy_index]
                        pos_actions[negative] = pos_actions[copy_index]
                        pos_prev_actions[negative] = pos_prev_actions[copy_index]
                        if pos_rews[copy_index] > 0:
                            true_pos += 1
                        else:
                            true_neg += 1
                        positive +=1
                        wasted +=1
                    elif wasted > waste_limit: # no more tolerance, so we just pretend the sample was positive
                        pos_observations[positive] = result["ob"][i]
                        pos_true_rews[positive] = result["true_rew"][i]
                        pos_rews[positive] = result["rew"][i]
                        pos_vpreds[positive] = result["vpred"][i]
                        pos_dones[positive] = result["dones"][i]
                        pos_actions[positive] = result["ac"][i]
                        pos_prev_actions[positive] = result["prevac"][i]
                        true_neg +=1
                        positive +=1                        
                    else:
                        wasted +=1
                if negative + positive == horizon:
                    break

        print("AAActor episode: {} positive, {} negative, {} wasted".format(true_pos, true_neg, wasted))

        observations = np.concatenate((pos_observations, neg_observations))
        rews = np.concatenate((pos_rews, neg_rews))
        dones = np.concatenate((pos_dones, neg_dones))
        true_rews = np.concatenate((pos_true_rews, neg_true_rews))
        vpreds = np.concatenate((pos_vpreds, neg_vpreds))
        actions = np.concatenate((pos_actions, neg_actions))
        prev_actions = np.concatenate((pos_prev_actions, neg_prev_actions))
        
        yield {"ob": observations, "rew": rews, "dones": dones, "true_rew": true_rews, "vpred": vpreds,
               "ac": actions, "prevac": prev_actions, "nextvpred": nextvpreds, "ep_rets": ep_rets,
               "ep_lens": ep_lens, "ep_true_rets": ep_true_rets, "total_timestep": total_timesteps}

def balanced_sample(seg_gen, horizon, gamma, lam, waste_limit=10000):
    assert horizon % 2 == 0, "Horizon {} should be divisible by two".format(horizon)
    half_size = horizon // 2

    observations = []
    dones = []
    true_rews = []
    vpreds = []
    actions = []
    advs = []
    tdlamrets = []
    ep_rets = []
    ep_lens = []
    total_timesteps = 0

    pos = 0
    neg = 0
    wasted = 0

    while pos + neg < horizon:    
        seg = seg_gen.__next__()
        add_vtarg_and_adv(seg, gamma, lam)

        total_timesteps += seg["total_timestep"]
        ep_rets += seg["ep_rets"]
        ep_lens += seg["ep_lens"]

        for i in range(horizon):
            if pos + neg == horizon:
                break
            
            isPositive = seg["tdlamret"][i] > 0
            if isPositive and (pos < half_size or wasted >= waste_limit):
                pos += 1
                add = True
            elif not isPositive and (neg < half_size or wasted >= waste_limit):
                neg += 1
                add = True
            else:
                wasted += 1
                add = False

            if add:
                observations.append(copy.deepcopy(seg["ob"][i]))
                dones.append(copy.deepcopy(seg["dones"][i]))
                true_rews.append(copy.deepcopy(seg["true_rew"][i]))
                vpreds.append(copy.deepcopy(seg["vpred"][i]))
                actions.append(copy.deepcopy(seg["ac"][i]))
                advs.append(copy.deepcopy(seg["adv"][i]))
                tdlamrets.append(copy.deepcopy(seg["tdlamret"][i]))

    observations = np.array(observations)
    dones = np.array(dones)
    true_rews = np.array(true_rews)
    vpreds = np.array(vpreds)
    actions = np.array(actions)
    advs = np.array(advs)
    tdlamrets = np.array(tdlamrets)
    print("AAActor episode: {} positive, {} negative, {} wasted".format(pos, neg, wasted))
    
    return {
        "ob": observations,
        "dones": dones,
        "true_rew": true_rews,
        "vpred": vpreds,
        "ac": actions,
        "ep_rets": ep_rets,
        "ep_lens": ep_lens,
        "total_timestep": total_timesteps,
        "adv": advs,
        "tdlamret": tdlamrets
    }
