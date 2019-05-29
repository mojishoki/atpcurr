from ctypes import *
import time
import numpy as np

from util import ClausePrinter

#library_path = "./fcoplib_20181109.so"
#library_path = "./fcoplib_20181130.so"
#library_path = "./fcoplib_20181221.so"
library_path = "./fcoplib_20190513.so"

cop_lib = cdll.LoadLibrary(library_path)
cop_lib.cop_caml_init()
cop_start = cop_lib.py_start
cop_start.restype = py_object
cop_start.argtypes = [c_char_p]
cop_action = cop_lib.py_action
cop_action.restype = py_object
cop_backtrack = cop_lib.py_backtrack
cop_backtrack.restype = None

cop_restart = cop_lib.py_restart
cop_restart.restype = py_object

cop_st_print = cop_lib.py_st_print
cop_st_print.restype = None

cop_st_features = cop_lib.py_st_features
cop_st_features.restype = py_object
cop_st_features_fast = cop_lib.py_st_features_fast
cop_st_features_fast.restype = py_object

cop_contr_features = cop_lib.py_contr_features
cop_contr_features.restype = py_object
cop_contr_represent = cop_lib.py_contr_represent
cop_contr_represent.restype = py_object

cop_st_represent = cop_lib.py_st_represent
cop_st_represent.restype = py_object

cop_all_contras = cop_lib.py_all_contras
cop_all_contras.restype = py_object
cop_nos_contras = cop_lib.py_nos_contras
cop_nos_contras.restype = py_object

cp = ClausePrinter()

class Backend(object):
    def __init__(self, indexer, verbose=True, fast_features=True, use_previous_action=False):
        self.indexer = indexer
        self.verbose = verbose
        self.fast_features = fast_features
        self.use_previous_action = use_previous_action
        self.cop_action_time = 0
        self.cop_backtrack_time = 0
        self.cop_nos_contras_time = 0
        self.cop_st_features_time = 0
        self.cop_contr_features_time = 0
        self.state_time = 0
        self.action_time = 0
    def start(self, file, pathLim=0):
        (done, action_count) = cop_start(file.encode('UTF-8'))
        return self.process(done, action_count)
    def restart(self, pathLim=0):
        (done, action_count) = cop_restart()
        return self.process(done, action_count)
    def step(self, action_index):
        t1 = time.time()
        action_index = int(action_index)
        (done, action_count) = cop_action(action_index)
        self.cop_action_time += time.time() - t1
        result = self.process(done, action_count, action_index)
        return result
    def backtrack(self):
        if self.use_previous_action:
            assert False, "Backtrack does not update previous_actions"
        t1 = time.time()
        cop_backtrack()
        self.cop_backtrack_time += time.time() - t1
    def hash_to_index(self, hash):
        t1 = time.time()
        result = cop_nos_contras().index(hash)
        self.cop_nos_contras_time += time.time() - t1
        return result
    def index_to_hash(self, index):
        t1 = time.time()
        result = cop_nos_contras()[index]
        self.cop_nos_contras_time += time.time() - t1
        return result
    def print_state(self):
        cop_st_print()
        # rep = cop_st_represent()
        # print("\n-----------")
        # print("Holstep format: ", rep)
        # print("Holstep goal: ", cp.stringify(rep[0]))
        # print("Holstep path: ", cp.stringify(rep[1]))
        # print("Holstep lemmas: ", cp.stringify(rep[2]))
        # print("Holstep moregoals: ", cp.stringify(rep[3]))
        # print("Extra features: ", cp.get_leftright_symbols(rep[0], ("mul","plus")))
        # print("-----------\n")

    def process(self, done, action_count, previous_action_index=None):
        if self.verbose: cop_st_print()
        if done != 1 and action_count == 0:
            done = -1
        t1 = time.time()
        if self.fast_features:
            goal = cop_st_features_fast()
            path = []
            lem = []
            moregoals = []
        else:
            (goal, path, lem, moregoals) = cop_st_features()
        t2 = time.time()
        
        action_list=[]
        for i in range(action_count):
            action = cop_contr_features(i)
            action_list.append(action)
        t3 = time.time()


        if self.use_previous_action:
            if previous_action_index is not None:
                previous_action = self.previous_actions[previous_action_index]
            else:
                previous_action = []
            _, previous_embedded_action = self.indexer.store_action(previous_action)
            self.previous_actions = action_list
        
        embedded_actions = [self.indexer.store_action(action) for action in action_list]
        embedded_action_ids = [a[0] for a in embedded_actions]
        embedded_action_list = [a[1] for a in embedded_actions]
        t4 = time.time()

        if done == 0:
            state_id, state = self.indexer.store_state(goal, path, lem, moregoals)
            if self.use_previous_action:
                state = np.insert(state, len(state), previous_embedded_action, axis=0)
        elif done == 1:
            state_id, state = 0, "success"
        elif done == -1:
            state_id, state = 0, "failure"
        t5 = time.time()
        
        self.cop_st_features_time += t2 - t1
        self.cop_contr_features_time += t3 - t2
        self.action_time += t4 - t3
        self.state_time += t5 - t4
        
        if self.verbose: print("state: ", state_id)
        return state, state_id, embedded_action_list, embedded_action_ids, done
