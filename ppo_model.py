import numpy as np
import tensorflow as tf
from stable_baselines.common.policies import ActorCriticPolicy, nature_cnn
from stable_baselines.a2c.utils import conv, linear, conv_to_fc, batch_to_seq, seq_to_batch, lstm
from stable_baselines.common.distributions import make_proba_dist_type
from stable_baselines.common.input import observation_input
from baselines import logger

from util import AttrDict

class FeedForwardPolicy(ActorCriticPolicy):
    """
    Policy object that implements actor critic, using a feed forward neural network.
    :param sess: (TensorFlow session) The current TensorFlow session
    :param ob_space: (Gym Space) The observation space of the environment
    :param ac_space: (Gym Space) The action space of the environment
    :param n_env: (int) The number of environments to run
    :param n_steps: (int) The number of steps to run for each environment
    :param n_batch: (int) The number of batch to run (n_envs * n_steps)
    :param reuse: (bool) If the policy is reusable or not
    :param layers: ([int]) The size of the Neural network for the policy (if None, default to [64, 64])
    :param cnn_extractor: (function (TensorFlow Tensor, ``**kwargs``): (TensorFlow Tensor)) the CNN feature extraction
    :param feature_extraction: (str) The feature extraction type ("cnn" or "mlp")
    :param kwargs: (dict) Extra keyword arguments for the nature CNN feature extraction
    """

    def __init__(self, sess, ob_space, ac_space, n_env, n_steps, n_batch, args=None, layers=None, reuse=False,
                 cnn_extractor=nature_cnn, feature_extraction="cnn", **kwargs):
        super(FeedForwardPolicy, self).__init__(sess, ob_space, ac_space, n_env, n_steps, n_batch, n_lstm=256,
                                                reuse=reuse, scale=(feature_extraction == "cnn"))
        self.args = args
        self.step_counter = 0
                    

        action_indices = list(range(self.args.n_action_slots))
        state_indices = [x+self.args.n_action_slots for x in range(self.args.state_dim)]
        with tf.variable_scope("model", reuse=reuse):
            if self.args.value_gets_actions:
                if self.args.actions_added:
                    pi_latent, vf_latent, value_fn = build_actor_critic_network_actionsadded(self.processed_x, layers, action_indices, state_indices, reuse)
                else:
                    pi_latent, vf_latent, value_fn = build_actor_critic_network_tri(self.processed_x, layers, action_indices, state_indices, reuse)
            else:
                pi_latent, vf_latent, value_fn = build_actor_critic_network_tri_separate_vf(self.processed_x, layers, action_indices, state_indices, args.latent_dim)

            self.policy = pi_latent
            self.proba_distribution = self.pdtype.proba_distribution_from_flat(pi_latent)
            self.q_value = vf_latent

            # self.proba_distribution, self.policy, self.q_value = \
            #     self.pdtype.proba_distribution_from_latent(pi_latent, vf_latent, init_scale=0.01)


        self.value_fn = value_fn
        self.initial_state = None
        self._setup_init()

    def step(self, obs, state=None, mask=None, deterministic=False):
        self.step_counter += 1
        if deterministic:
            action, value, neglogp = self.sess.run([self.deterministic_action, self._value, self.neglogp],
                                                   {self.obs_ph: obs})
        else:
            # action, value, neglogp = self.sess.run([self.action, self._value, self.neglogp],
            action, value, neglogp, policy_np = self.sess.run([self.action, self._value, self.neglogp, self.policy],
                                                   {self.obs_ph: obs})
            policy_np = policy_np[0]
            policy_np = policy_np[policy_np > -1e5]
            logger.logkv("update_no", self.step_counter)
            logger.record_tabular("policy_logit_avg", np.mean(policy_np))

            #print("policy ", policy_np, action)
        return action, value, self.initial_state, neglogp

    def proba_step(self, obs, state=None, mask=None):
        return self.sess.run(self.policy_proba, {self.obs_ph: obs})

    def value(self, obs, state=None, mask=None):
        return self.sess.run(self._value, {self.obs_ph: obs})


def build_actor_critic_network(x, layers, num_actions, num_state, reuse):
    activ = tf.nn.relu
    pis = []
    vfs = []
    x = tf.layers.flatten(x)
    for i in range(num_actions + num_state):
        with tf.variable_scope("actor_critic", reuse=tf.AUTO_REUSE):
            x_prime = x[:, i, :]
            pi_h = x_prime
            vf_h = x_prime
            for i, layer_size in enumerate(layers):
                pi_h = activ(linear(pi_h, 'pi_fc' + str(i), n_hidden=layer_size, init_scale=np.sqrt(2)))
                vf_h = activ(linear(vf_h, 'vf_fc' + str(i), n_hidden=layer_size, init_scale=np.sqrt(2)))
            pis.append(pi_h)
            vfs.append(vf_h)
    pi_h = tf.layers.flatten(tf.concat(pis, axis=1))
    vf_h = tf.layers.flatten(tf.concat(vfs, axis=1))
    pi_latent = pi_h
    vf_latent = vf_h
    value_fn = linear(vf_h, 'vf', 1)
    return pi_latent, vf_latent, value_fn

# Peasant method: linear (num_actions) layer for head of actions on top of weight shared latent embeddings
def build_actor_critic_network_peasant_method(x, layers, action_indices, state_indices, reuse):
    activ = tf.nn.relu
    pis = []
    vfs = []
    #x = tf.layers.flatten(x)
    for i in range(len(action_indices) + len(state_indices)):
        with tf.variable_scope("actor_critic", reuse=tf.AUTO_REUSE):
            x_prime = x[:, i, :]
            pi_h = x_prime
            vf_h = x_prime
            for j, layer_size in enumerate(layers):
                pi_h = activ(linear(pi_h, 'pi_fc' + str(j), n_hidden=layer_size, init_scale=np.sqrt(2)))
                vf_h = activ(linear(vf_h, 'vf_fc' + str(j), n_hidden=layer_size, init_scale=np.sqrt(2)))
            pis.append(pi_h)
            vfs.append(vf_h)
    pi_h = tf.layers.flatten(tf.concat(pis, axis=1))
    vf_h = tf.layers.flatten(tf.concat(vfs, axis=1))
    pi_latent = linear(pi_h, 'pi_head', len(action_indices))
    vf_latent = linear(vf_h, 'vf_head', len(action_indices))
    value_fn = linear(vf_latent, 'vf', 1)
    return pi_latent, vf_latent, value_fn


# num_actions x (a, s1, s2)
def build_actor_critic_network_tri(x, layers, action_indices, state_indices, reuse):
    activ = tf.nn.relu
    vfs = []
    #x = tf.layers.flatten(x)
    for i in action_indices:
        ind = [i]
        ind.extend(state_indices)
        ind = np.array(ind, dtype=np.int32)
        with tf.variable_scope("actor_critic", reuse=tf.AUTO_REUSE):
            x_prime = tf.layers.flatten(tf.gather(x, ind, axis=1))
            vf_h = x_prime
            for j, layer_size in enumerate(layers):
                vf_h = activ(linear(vf_h, 'vf_fc' + str(j), n_hidden=layer_size, init_scale=np.sqrt(2)))
            vf_h = activ(linear(vf_h, 'vf_fc_last', n_hidden=10, init_scale=np.sqrt(2)))
            vfs.append(vf_h)
    vf_h = tf.layers.flatten(tf.concat(vfs, axis=1))
    vf_latent = activ(linear(vf_h, 'vf_head', len(action_indices)))
    value_fn = linear(vf_latent, 'vf', 1)

    pi_latent = build_policy(x, layers, action_indices, state_indices, activ)
    
    return pi_latent, vf_latent, value_fn

def build_actor_critic_network_actionsadded(x, layers, action_indices, state_indices, reuse):
    activ = tf.nn.relu
    with tf.variable_scope("actor_critic", reuse=tf.AUTO_REUSE):
        actions = tf.gather(x, action_indices, axis=1)
        actions = tf.reduce_sum(actions, axis=1, keepdims=True)
        state = tf.gather(x, state_indices, axis=1)
        vf_h = tf.layers.flatten(tf.concat([actions, state], axis=1))
        for j, layer_size in enumerate(layers):
            vf_h = activ(linear(vf_h, 'vf_fc' + str(j), n_hidden=layer_size, init_scale=np.sqrt(2)))
    vf_latent = activ(linear(vf_h, 'vf_head', len(action_indices)))
    value_fn = linear(vf_latent, 'vf', 1)

    pi_latent = build_policy(x, layers, action_indices, state_indices, activ)
    
    return pi_latent, vf_latent, value_fn

# policy uses towers on (action_i, goal, path) triples
# vf uses only (goal, path)
def build_actor_critic_network_tri_separate_vf(x, layers, action_indices, state_indices, latent_dim):
    print("policy works on triples, value works on state only")
    activ = tf.nn.relu

    if latent_dim is not None:
        x = tf.layers.conv1d(x, filters=latent_dim, kernel_size=1, strides=1, padding="valid", data_format="channels_last", name="toLatent")

    # value function (works from state only)
    vf_h = tf.layers.flatten(tf.gather(x, state_indices, axis=1))
    for i, layer_size in enumerate(layers):
        vf_h = activ(linear(vf_h, 'vf_fc' + str(i), n_hidden=layer_size, init_scale=np.sqrt(2)))
    vf_latent = vf_h
    value_fn = linear(vf_h, 'vf', 1)

    pi_latent = build_policy(x, layers, action_indices, state_indices, activ)

    return pi_latent, vf_latent, value_fn

def build_policy(x, layers, action_indices, state_indices, activ):
    # policy function (works on (action_i, goal, path) triples
    pis = []
    for i in action_indices:
        ind = [i]
        ind.extend(state_indices)
        ind = np.array(ind, dtype=np.int32)
        with tf.variable_scope("actor_critic", reuse=tf.AUTO_REUSE):
            x_prime = tf.layers.flatten(tf.gather(x, ind, axis=1))
            pi_h = x_prime
            for j, layer_size in enumerate(layers):
                pi_h = activ(linear(pi_h, 'pi_fc' + str(j), n_hidden=layer_size, init_scale=np.sqrt(2)))
            pi_h = linear(pi_h, 'pi_fc_last', n_hidden=1, init_scale=np.sqrt(2))

            ind_action = np.array([i], dtype=np.int32)
            action_sum = tf.reduce_sum(tf.gather(x, ind_action, axis=1))
            pi_h = tf.cond(action_sum > 0, lambda: pi_h, lambda: pi_h * 0 - 1e7)

            pis.append(pi_h)
    pi_latent = tf.layers.flatten(tf.concat(pis, axis=1))
    # pi_latent = linear(pi_h, 'pi_head', len(action_indices))
    return pi_latent
    
