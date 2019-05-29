#!/usr/bin/python
from ctypes import *
import time
import argparse
import os
import util
import json
import re
import os
import errno
import numpy as np

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--binary', help='binary file of the ocaml leancop prover', type=str, default='./fcoplib_20181109.so')
parser.add_argument('--time', help='time in seconds, floats are OK', type=float, default=10.)
parser.add_argument('--file', help='file name with the theorem to be proved', type=str, default='theorems/a.p')
parser.add_argument('--directory', help='directory containing problems', type=str, default=None)
parser.add_argument('--proof_format', help='feature/holstep/hash/index/index_by_hash', type=str, default='hash')
parser.add_argument('--write_proof_files', help='whether to write found proofs to files', type=bool, default=False)
parser.add_argument('--feature_file', help='save used features into this file', type=str, default='feature_list.npy')
parser.add_argument('--nonzero_var_mask', help='output file in which we save a binary mask that shows nonezero variance features', type=str, default='nonzero_var_mask.csv')
args = parser.parse_args()

cop_lib = cdll.LoadLibrary(args.binary)
cop_lib.cop_caml_init()
cop_start = cop_lib.py_start
cop_start.restype = py_object
cop_start.argtypes = [c_char_p]
cop_action = cop_lib.py_action
cop_action.restype = py_object
cop_backtrack = cop_lib.py_backtrack
cop_backtrack.restype = None

cop_contr_features = cop_lib.py_contr_features
cop_contr_features.restype = py_object
cop_contr_represent = cop_lib.py_contr_represent
cop_contr_represent.restype = py_object

cop_st_print = cop_lib.py_st_print
cop_st_print.restype = None

cop_st_features = cop_lib.py_st_features
cop_st_features.restype = py_object
cop_all_contras = cop_lib.py_all_contras
cop_all_contras.restype = py_object
cop_nos_contras = cop_lib.py_nos_contras
cop_nos_contras.restype = py_object

cop_restart = cop_lib.py_restart
cop_restart.restype = py_object

S_DIM = 60
A_DIM = 30
indexer = util.Indexer(S_DIM, A_DIM)
cp = util.ClausePrinter()

SKIP_FIRST = False
MIN_STEPS = 0 # start iterative deepening at this depth


def prove(moresteps, acs, t0, timelimit):
    def doact(i):        
        (p, nacs) = cop_action(i)
        prove.prf.append(i)
        #print(prove.prf)
        prove.actions += 1
        if (p == 1 or (p == 0 and moresteps > 0 and time.time() - t0 < args.time and prove(moresteps - 1, nacs, t0, timelimit))):
            return True
        prove.prf.pop()
        cop_backtrack()
        return False

    # store features
    for i in range(acs):
        indexer.store_action(cop_contr_features(i))
    state = cop_st_features()
    indexer.store_state(state[0], state[1])

    
    if (-2133191988894931610 in cop_nos_contras()): # prod-s
        if doact(cop_nos_contras().index(-2133191988894931610)):
            return True
    elif (-1857960034002651292 in cop_nos_contras()): # sum-s
        if doact(cop_nos_contras().index(-1857960034002651292)):
            return True
    elif (3834872697031308342 in cop_nos_contras()): # s=s
        if doact(cop_nos_contras().index(3834872697031308342)):
            return True
    elif (4122309023149547824 in cop_nos_contras()): # sum-o
        if doact(cop_nos_contras().index(4122309023149547824)):
            return True
    elif (-1929448961502595998 in cop_nos_contras()): # prod-o
        if doact(cop_nos_contras().index(-1929448961502595998)):
            return True
    else:
    
    # if True:
        for i in range(0,acs):
            # ... => sum = sum
            # ... => prod = prod
            if cop_nos_contras()[i] != -1916784989292842441 \
              and cop_nos_contras()[i] != 3631082610621495653 \
              and cop_nos_contras()[i] != 3559362369102099298 \
              and doact(i):
                return True
        return False



def find_one_proof(file, timelimit):
    t0 = time.time()
    cop_start(file.encode('UTF-8'))

    success = False
    maxsteps = MIN_STEPS
    prove.actions = 0
    while True:
        maxsteps += 1
        if time.time() - t0 >= timelimit:
            break
        (_, macs) = cop_restart()
        if SKIP_FIRST:
            (_, macs) = cop_action(0) # TODO disregard the first deterministic step            
        prove.prf = []
        print("Checking length {}".format(maxsteps))
        if prove(maxsteps, macs, t0, timelimit):
            print("proof found")
            success = True
            if args.proof_format == "index":
                nprf = prove.prf
            else:
                nprf = []
                cop_restart()
                if SKIP_FIRST:
                    cop_action(0) # TODO disregard the first deterministic step
                for i in prove.prf:
                    if args.proof_format == "hash":
                        nprf.append(cop_nos_contras()[i])
                    elif args.proof_format == "index_by_hash":
                        step = cop_nos_contras()[i]
                        if step in util.ACTION_MAP.keys():
                            step = util.ACTION_MAP[step]
                        nprf.append(step)
                    elif args.proof_format == "feature":
                        nprf.append(cop_contr_features(i))
                    elif args.proof_format == "holstep":
                        cop_st_print()
                        step = cop_contr_represent(i)
                        #step = cp.stringify(step)
                        nprf.append(step)
                    cop_action(i)
            if args.write_proof_files:
                outfile = get_output_file_name(file, args.binary, ".proof")
                make_sure_path_exists(outfile)
                print("outfile: ", outfile)
                
                with open(outfile, 'w') as outfile:
                    nprf_rev = nprf[:]
                    json.dump(nprf_rev, outfile)

            print ("Proof found at depth {}, time: {} sec, actions tried {}, format: {}\n{}".format(maxsteps, time.time() - t0, prove.actions, args.proof_format, nprf))
            break

    if not success:
        print("Failed to find proof in {} sec, reaching depth {}".format(args.time, maxsteps))

def get_output_file_name(file, binary, suffix=""):
    m = re.search(r'(?<=/)\w+(?=.so)', binary)
    binary_version = m.group(0)
    last_slash = file.rfind("/")
    outfile = file[:last_slash] + "/{}/".format(binary_version) + file[last_slash+1:] + suffix
    return outfile

def make_sure_path_exists(filename):
    dir = os.path.dirname(filename)
    if not os.path.exists(dir):
        try:
            os.makedirs(dir)
        except OSError as exception:
            if exception.errno != errno.EEXIST:
                raise

        
if args.directory != None:
    dirs = args.directory.split(',')
    for dir in dirs:
        for filename in os.listdir(dir):
            if filename.endswith(".p"):
                # if not filename.endswith(".proof"):
                name = os.path.join(dir, filename)
                print("\n\nTrying to find proof for {}".format(name))
                find_one_proof(name,args.time)
else:
    find_one_proof(args.file, args.time)

# get the set of all features (goal, path, action) encountered during the proofs
goal_features = np.concatenate(indexer.goal_list)
path_features = np.concatenate(indexer.path_list)
action_features = np.concatenate(indexer.action_list)
found_features = np.concatenate((goal_features, path_features, action_features))

index_by_hash_features = np.array(list(util.FEATURE_MAP.keys())) # nasty hack

features = np.concatenate((found_features, index_by_hash_features))

unique_features = np.unique(features)
unique_features = np.sort(unique_features).astype(int)

np.save(args.feature_file, unique_features)
print("Total features collected: {}".format(unique_features.shape))

    
# actions, mask = indexer.get_dense_actions(variance_threshold=None)
# print("Action space:", actions.shape)

# actions2, mask2 = indexer.get_dense_actions(variance_threshold=0.)
# print("Action space after removing zero variance features:", actions2.shape)

# csv_text = ",".join(mask2.astype(int).astype(str))
# with open(args.nonzero_var_mask, 'w') as f:
#     f.write(csv_text)

