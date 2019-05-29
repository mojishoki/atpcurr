import numpy as np
import time
from ctypes import *
import argparse
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--backend', help='ocaml/prolog', type=str, default='prolog')
parser.add_argument('--ocaml_version', help='hash/literal', type=str, default='literal')
parser.add_argument('--time', help='time in seconds, floats are OK', type=float, default=10.)
parser.add_argument('--file', help='file name with the theorem to be proved', type=str, default='theorems/a.p')
args = parser.parse_args()

import util
import backend_ocaml
import env_prolog

S_DIM = 60
A_DIM = 30

indexer = util.Indexer(S_DIM, A_DIM)
cp = util.ClausePrinter()

if args.backend == "ocaml":
    env = backend_ocaml.Backend(indexer, verbose=False)
elif args.backend == "prolog":
    env = env_prolog.Env(indexer, backtrack=True, verbose=False, print_proof=True)

t0 = time.time()
def getTime():
    return time.time() - t0

pathLim = 1
while getTime() < args.time:
    state, state_id, action_list, action_ids, done = env.start(args.file, pathLim)

    while done == 0 and getTime() < args.time:
        action_index = 0
        state, state_id, action_list, action_ids, done = env.step(action_index)

    if done == 1: # proof has been found
        print("Proof found in {} sec at depth {}.".format(getTime(), pathLim))
        break
    elif done == -1:
        print("Proof failed in {} sec at depth {}.".format(getTime(), pathLim))
        pathLim += 1
        if args.backend == "ocaml": # iterative deepening is only implemented for prolog now
            break

    
if done != 1:
    print("Could not find proof in {} sec".format(args.time))
