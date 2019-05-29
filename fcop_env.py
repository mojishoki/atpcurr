import numpy as np
import os
import sys
import random
import json
from pathlib import Path
import gym
from gym import spaces
from gym.spaces import Discrete, MultiDiscrete, MultiBinary, Box
from baselines import logger
from ctypes import *
import gc

from backend_ocaml import Backend
import util

np.set_printoptions(threshold=np.nan)

class StageScheduler:
  def __init__(self, args):
    self.args = args
    self.steps_to_remain = args.scheduler_starting_step
    self.last_timesteps_so_far = 1
    self.consecutive_dones = 0
    self.locals = {"timesteps_so_far":0}
    self.progress = 0.0
    
  def on_actor_batch_begin(self, locals, globals):
    self.locals = locals
    self.globals = globals
    self.progress = locals["timesteps_so_far"] * 1.0 / locals["total_timesteps"]
    sys.stdout.flush()

    if locals["timesteps_so_far"] >= (self.last_timesteps_so_far + self.args.steps_per_curriculum):
      self.set_steps_to_remain(self.steps_to_remain + 1)

    if self.total_episodes > 0:
      pct = 1.0 * self.total_dones / self.total_episodes
      print("DONE PERCENTAGE: {}/{} = {} ".format(self.total_dones, self.total_episodes, pct))
      logger.logkv("total_dones", self.total_dones)
      logger.logkv("total_episodes", self.total_episodes)
      logger.logkv("done_percentage", pct)
      if self.args.quick_progress_percentage is not None:
        if pct > self.args.quick_progress_percentage:
          self.consecutive_dones = 0
          self.set_steps_to_remain(self.steps_to_remain+1)
          
    self.total_dones = 0
    self.total_episodes = 0

  def on_reset(self):
    pass


  def on_episode_over(self, done, dict, file):
    if dict["proof"] is not None:
      self.total_episodes += 1
      if done == 1:
        self.total_dones += 1
        self.consecutive_dones += 1
      else:
        self.consecutive_dones = 0

    if self.args.quick_progress_threshold is not None:
      if self.consecutive_dones >= self.args.quick_progress_threshold:
        self.consecutive_dones = 0
        self.set_steps_to_remain(self.steps_to_remain+1)

    # update problem specific steps_to_remain
    if self.args.scheduler_type == "local" and dict["episodes"] >= 50:
      if dict["episodes"] > self.args.steps_per_curriculum: # forward the curriculum no matter what
        dict["steps_to_remain"] += 1
        print("LOCAL curriculum: {} - {}".format(file, dict["steps_to_remain"]))
        logger.logkv("steps_to_remain", dict["steps_to_remain"])
      elif self.args.quick_progress_percentage is not None: # fast forward allowed
        pct = 1.0 * dict["success"] / dict["episodes"]
        if self.args.quick_progress_percentage < pct:
          dict["steps_to_remain"] += 1
          print("LOCAL (accelerated) curriculum {} - {}".format(file, dict["steps_to_remain"]))
          logger.logkv("steps_to_remain", dict["steps_to_remain"])

      dict["success"] = 0
      dict["episodes"] = 0

  def on_env_init(self):
    self.consecutive_dones = 0
    self.total_dones = 0
    self.total_episodes = 0

  def set_steps_to_remain(self, steps_to_remain):
    if self.args.scheduler_type == "global":
      self.steps_to_remain = steps_to_remain
      self.last_timesteps_so_far = self.locals["timesteps_so_far"]
      logger.logkv("steps_to_remain", steps_to_remain)
      print("GLOBAL curriculum: ", steps_to_remain)


class Container:
    def __init__(self, args, stage_scheduler):
        self.locals = None
        self.globals = None
        self.args = args
        self.stage_scheduler = stage_scheduler
        self.model = None

    def callback(self, locals, globals):
        self.locals = locals
        self.globals = globals

        # self.model.save("{}/ppo1_fcop".format(self.args.outdir))

        self.stage_scheduler.on_actor_batch_begin(locals, globals)


    
class BasicEnv(gym.Env):
  def __init__(self, args, container):
    super().__init__()
    self.args = args
    self.container = container

    self.indexer = util.Indexer(args.n_dim, args.n_dim, feature_file=args.feature_file, embed_separately=True)
    self.n_dim = self.indexer.S_DIM    
    self.backend = Backend(self.indexer, verbose=False, fast_features=args.fast_features, use_previous_action=args.use_previous_action)
    self.action_space = spaces.Discrete(args.n_action_slots)
    self.observation_space = spaces.Box(low=-10.0, high=10.0, shape=(args.n_action_slots+args.state_dim, self.n_dim), dtype=np.float32)

    self.last_file = None


  def set_file(self, file):
    self.file = file
    self.reset()

  def reset(self):
    assert "file" in dir(self), "First call BasicEnv.set_file(file)!!!"
    gc.collect()
    
    if self.file == self.last_file:
      self.env_dict = self._process(self.backend.restart())
    else:
      self.env_dict = self._process(self.backend.start(self.file))
      self.last_file = self.file
    self.env_history = []
    self.real_steps = []
    self.step_counter = 0
    self.obs_state = self._construct_obs_state(self.env_dict)
     
    return self.obs_state

  # like step, but there is no bookkeeping, no checking for illegal steps
  # no side effect
  def pure_step(self, action):
    env_dict = self._process(self.backend.step(int(action)))
    obs_state = self._construct_obs_state(env_dict)
    done = env_dict["done"]
    if done == 1:
      episode_over = True
      reward = 1.0
    elif done == -1:
      episode_over = True
      reward = self.args.failure_reward
    else:
      episode_over = False
      reward = 0
    return obs_state, reward, episode_over, env_dict
    

  def step(self, action):
    self.step_counter += 1
      
    if action >= self.env_dict["action_count"]:
      reward = self.args.illegal_reward
      episode_over = self.args.terminate_on_illegal
      print("ILLEGAL")
    else:
      self.real_steps.append(action)
      self.obs_state, reward, episode_over, self.env_dict = self.pure_step(action)
      self.env_history.append(self.env_dict)
      
    return self.obs_state, reward, episode_over, self.env_dict

  def backtrack(self):
    self.backend.backtrack()
    self.real_steps.pop()
    self.env_dict = self.env_history.pop()

  # this returns a dictionary that fully describes the proof state (so it can be restored)
  # no side effect
  def _process(self, t):
    state, _state_id, action_list, _action_ids, done = t
    action_count = len(action_list)
    env_dict = {"state":state, "action_list":action_list, "action_count":action_count, "done": done}
    return env_dict

  # no side effect
  def _construct_obs_state(self, env_dict):
    obs_state = env_dict["action_list"][:self.args.n_action_slots]
    for k in range(env_dict["action_count"], self.args.n_action_slots):
        obs_state.append(np.zeros(self.n_dim,))
    obs_state_np = np.array(obs_state)

    state = env_dict["state"]
    if  state is "success" or state is "failure":
      state = np.zeros((self.args.state_dim, self.n_dim))
    elif state.ndim != 2:
      state = np.expand_dims(state, axis=0)
    obs_state_np = np.concatenate([obs_state_np, state])

    return obs_state_np


class DirectoryEnv(BasicEnv):
  def __init__(self, args, container):
    super().__init__(args, container)
    self.file_generator = self._gen_files()

  def set_directory(self, directory):
    print("setting directory to ", directory)
    self.directory = directory
    self.file_generator = self._gen_files()
    self.reset()

  def reset(self):
    self.file = next(self.file_generator)
    return super(DirectoryEnv,self).reset()

  def _gen_files(self):
    while True:
      dir_given = "directory" in dir(self)
      file_given = "file" in dir(self)
      assert dir_given or file_given, "First call BasicEnv.set_file(file) or DirectoryEnv.set_directory(directory)!!!"
      if dir_given:
        filenames = sorted(os.listdir(self.directory))
        filenames = list(filter(lambda x: x.endswith(".p"), filenames))
        if "train_file_count" in self.args:
          filenames = filenames[:self.args.train_file_count]
        for filename in filenames:
          filepath = os.path.join(self.directory, filename)
          for _ in range(self.args.same_file_count):
            yield filepath
      elif file_given:
        yield self.file


class ProofEnv(DirectoryEnv):
  def __init__(self, args, container, stage_scheduler):
    super().__init__(args, container)
    self.stage_scheduler = stage_scheduler
    self.stage_scheduler.on_env_init()
    self.jackpots = []
    self.problemdict = {} # to store various useful things about proofs


  def reset(self):
    self.obs_state = super(ProofEnv,self).reset()
                  
    if self.file not in self.problemdict:
      self.problemdict[self.file] = {"success":0, "episodes":0, "proof":None, "steps_to_remain":self.args.scheduler_starting_step}

    dict = self.problemdict[self.file]
    proof_filepath = self._get_proof_filepath(self.file)
    if not self.args.proof_allowed: # no curriculum allowed
      self.proof_actions = None
    elif dict["proof"] is not None: # we have a stored proof, so we use it
      self.proof_actions = dict["proof"]
    elif proof_filepath is not None: # we load the proof from file
      with open(proof_filepath) as f:
        self.proof_actions = json.load(f)
        dict["proof"] = self.proof_actions
    else:
      self.proof_actions = None # no proof available

    if self.args.scheduler_type == "local":
      self.steps_to_remain = dict["steps_to_remain"]
    elif self.args.scheduler_type == "global":
      self.steps_to_remain = self.stage_scheduler.steps_to_remain
    if self.proof_actions is not None:
      max_random = max(0, min(len(self.proof_actions) - self.steps_to_remain, self.args.add_rnd_steps_to_remain))
      self.orig_steps_to_remain = self.steps_to_remain
      self.steps_to_remain += random.randint(0,max_random)
      self.obs_state, self.env_dict = self._make_steps_in_proof(self.proof_actions, self.steps_to_remain, self.obs_state, self.env_dict)
    return self.obs_state

  


  def step(self, action):
    obs_state, reward, episode_over, env_dict = super(ProofEnv,self).step(action)

    # do not allow long paths
    if self.args.max_exploration is not None:
      # current_limit = int(20 + self.stage_scheduler.progress * (self.args.max_exploration - 20))
      current_limit = self.args.max_exploration
      if self.step_counter >= current_limit:
        episode_over = True
    if (self.proof_actions is not None) and (self.step_counter >= self.orig_steps_to_remain + self.args.known_proof_max_exploration):
      episode_over = True


    # apply supervised reward
    if (env_dict["done"] == 0) and (self.proof_actions is not None) and (self.steps_to_remain >= len(self.real_steps)):
      remaining_proof = self.proof_actions[-self.steps_to_remain:]
      if (len(self.real_steps) <= len(remaining_proof)) and (action == remaining_proof[len(self.real_steps)-1]):
        reward += self.args.supervised_reward
      else:
        reward -= self.args.supervised_reward 

    dict = self.problemdict[self.file]
    if env_dict["done"] == 1:
      dict["success"] += 1
      if dict["proof"] is None:
        current_proof = self.real_steps
        old_length = 100000
      else:
        current_proof = dict["proof"][:-self.steps_to_remain] + self.real_steps
        old_length = len(dict["proof"])
      if len(current_proof) < old_length:
        print("JACKPOPOTPOTPOTPTOPOTPOTPOTPTOPTOPTO!!!!!")
        print("Found shorter proof: {} len {}, {}".format(self.file, len(current_proof), current_proof))
        if dict["proof"] is None or self.args.can_replace_proof:
          dict["proof"] = current_proof[:]
          if self.args.scheduler_type == "local" and not self.file in self.jackpots:
            dict["steps_to_remain"] = self.args.scheduler_starting_step
          elif not self.file in self.jackpots:
            self.stage_scheduler.set_steps_to_remain(self.args.scheduler_starting_step)
        if not self.file in self.jackpots:
          self.jackpots.append(self.file)
          print("Jackpot count: {}".format(len(self.jackpots)))
          logger.logkv("jackpots", len(self.jackpots))
          print("Jackpots: {}".format(self.jackpots))


          
    if episode_over:
      dict["episodes"] += 1
      self.stage_scheduler.on_episode_over(env_dict["done"], dict, self.file)
    self.obs_state = obs_state
    self.env_dict = env_dict
    return self.obs_state, reward, episode_over, self.env_dict
  
    
  def _get_proof_filepath(self, filepath):
    filename = os.path.basename(filepath)
    proof_filepath = "{}/{}.proof".format(self.args.proof_dir, filename)
    if os.path.isfile(proof_filepath):
      return proof_filepath
    else:
      return None

  # no side effect
  def _make_steps_in_proof(self, proof_actions, steps_to_remain, obs_state, env_dict):
    assert proof_actions is not None
    total_actions = len(proof_actions)
    steps_to_make = total_actions - steps_to_remain
    if steps_to_make > 0:
      for i in range(steps_to_make):
        obs_state, reward, episode_over, env_dict = self.pure_step(proof_actions[i])
    return obs_state, env_dict
