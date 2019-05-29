import numpy as np

import backend_ocaml
# import backend_prolog
import util

problem = "theorems/robinson/robinson_1m1m1__1/robinson_1m1m1__1.p"
proof_index = [0, 2, 0, 2, 0, 0, 0, 2, 0, 0, 1, 2, 0, 0, 2, 0, 0, 1, 2, 0, 0, 1, 0]
# problem = "theorems/extended/one_neg_one/extended_1n1__0.p"
# proof_index = [0, 2, 0, 2, 3, 3, 11, 0, 0, 0]

# problem = "theorems/robinson/robinson_1m2__2m1/robinson_1m2__2m1.p"
# proof_index =   [0, 3, 0, 2, 0, 2, 0, 0, 2, 0, 0, 2, 0, 0, 2, 0, 0, 0, 2, 0, 0, 0, 1, 2, 0, 2, 0, 0, 2, 0, 0, 2, 0, 0]
# proof_index = [0, 3, 0, 2, 0, 2, 0, 0, 2, 0, 0, 2, 0, 0, 2, 0, 0, 0, 1, 2, 0, 2, 0, 0, 2, 0, 0, 2, 0, 3, 0, 1, 0]
# proof_index = [0, 3, 0, 2, 0, 2, 0, 0, 1, 2, 0, 2, 0, 0, 2, 0, 2, 0, 0, 2, 0, 0, 1, 2, 0, 2, 0, 0, 2, 0, 0]
# proof_index = [0, 3, 0, 2, 0, 2, 0, 0, 2, 0, 0, 2, 0, 0, 2, 0, 4, 0, 0, 0, 0, 1, 2, 0, 2, 0, 0, 2, 0, 0, 2, 0, 0]

proof = proof_index
proof_format = 'index' # index/index_by_hash


N_DIM = 100

INVERSE_ACTION_MAP = {}
for k in util.ACTION_MAP.keys():
    v = util.ACTION_MAP[k]
    INVERSE_ACTION_MAP[v] = k


indexer = util.Indexer(N_DIM, N_DIM, embed_separately=True)
# backend = backend_prolog.Backend(indexer, verbose=False, n_dim=N_DIM)
backend = backend_ocaml.Backend(indexer, verbose=False, fast_features=False, use_previous_action=True)

def start():
    global steps, hashes, states, actions
    state, state_id, action_list, action_ids, done = backend.start(problem)
    # print("State: ", state)
    # print("Goal: {}\nPath: {}\nLem: {}\nTodos: {}\n".format(indexer.goal_list[state_id], indexer.path_list[state_id], indexer.lem_list[state_id], indexer.moregoals_list[state_id]))
    # for id in action_ids:
    #     print("Actions: {} - \n{}".format(id, indexer.action_list[id]))
    backend.print_state()
    print("======================================")
    steps = []
    hashes = []
    states = [state]
    actions = [action_list]
    return len(action_list), done

def step(st):
    global steps, hashes, states, actions
    if proof_format == "index":
        s = st
    elif len(action_list) == 1:
        s = 0
    elif proof_format =="index_by_hash":
        hash = INVERSE_ACTION_MAP[st]
        s = backend.hash_to_index(hash)
    # h = backend.index_to_hash(s)
    # if h in util.ACTION_MAP:
    #     h = util.ACTION_MAP[h]
    # print("action {}, hash {}".format(s, h))
    print("action {}".format(s))
    state, state_id, action_list, action_ids, done = backend.step(s)
    # print("State: ", state)
    # print("Goal: {}\nPath: {}\nLem: {}\nTodos: {}\n".format(indexer.goal_list[state_id], indexer.path_list[state_id], indexer.lem_list[state_id], indexer.moregoals_list[state_id]))
    # for id in action_ids:
    #     print("Actions: {} - \n{}".format(id, indexer.action_list[id]))
    backend.print_state()
    print("======================================")
    steps.append(s)
    # hashes.append(h)
    states.append(state)
    actions.append(action_list)
    return len(action_list), done


start()

for i, st in enumerate(proof):
    print("step ", i) 
    ac_count, done = step(st)
    print("Done: ", done)

    
print("Index sequence: ", steps)
# print("Hash sequence: ", hashes)

if False:
    xs = []
    ys = []
    for i, selected in enumerate(steps):
        index = steps[i]
        state = states[i]
        action_list = actions[i]

        for j, a in enumerate(action_list):
            x = np.concatenate([state,a])
            if j == index: # good step
                y = 1
            else:
                y = 0
            xs.append(x)
            ys.append(y)
    np.savez("sps.npz", x=np.array(xs), y=np.array(ys))
                
