import time
import numpy as np
import os
import sys

from stable_baselines import PPO1
from fcop_env import ProofEnv, StageScheduler, Container
from baselines import logger

from awarelib.neptune_utils import get_configuration

PRINT_PROBS = False
PROB_LIMIT = 1e-2 # discard actions with probability less than PROB_LIMIT
MIN_STEPS= 3000
MAX_STEPS= 3010
sys.setrecursionlimit(MAX_STEPS)

def safediv(x, y):
    if y == 0:
        return 0.0
    else:
        return x * 1.0 / y

def prove(model, env, moresteps, obs, t0):
    # print("\n\n")
    
    # print(env.real_steps)
    def doact(action_index):
        new_obs, reward, over, info = env.step(action_index)
        prove.actions += 1
        if (info["done"] == 1 or (info["done"] == 0 and moresteps > 0 and time.time() - t0 < args.evaltime and prove(model, env, moresteps - 1, new_obs, t0))):
            return True
        elif action_index > info["action_count"]:
            illegal_count += 1
            # print("Invalid({})".format(reward))
        else:
            env.backtrack()
            backtrack_count +=1
            # print("Backtrack({})".format(reward))
        return False

    probs = model.action_probability(obs)
    if PRINT_PROBS:
        print("probs: ", probs)
    prob_order = np.array(list(reversed(np.sort(probs))))
    action_order = np.array(list(reversed(np.argsort(probs))))
    action_order = action_order[prob_order > PROB_LIMIT]
    # print("Action order: ", action_order)
    # print("Prob order: ", prob_order)
    action_count = env.env_dict["action_count"] # TODO
    for i in action_order:
        if i >= action_count:
            # print("Skipping action {}".format(i))
            continue
        # env.backend.print_state()
        # print("Action {}/{} ".format(i, action_count-1))
        if doact(i):
            return True
    return False

def prove_nobacktrack(args, model, env, obs, t0):
    prove.actions = 0
    while True:
        prove.actions += 1
        t1 = time.time()
        probs = model.action_probability(obs)
        probs = probs[:env.env_dict["action_count"]]
        probs = probs / np.sum(probs)
        prove.guidance_time += time.time() - t1
        if args.evaltype == "det":
            action_index = np.argmax(probs)
        elif args.evaltype == "prob":
            action_index = np.random.choice(range(len(probs)), p=probs)
        else:
            assert False, "unknown evaltype {}".format(args.evaltype)
        if PRINT_PROBS:
            print(prove.actions, " probs:        ", probs)
            print(prove.actions, " action: ", action_index)
        obs, reward, over, info = env.step(action_index)
        if info["done"] == 1:
            return "success", prove.actions
        if info["done"] == -1:
            return "failure", 0
        if time.time() - t0 > args.evaltime:
            return "timeout", 0

        
def find_one_proof(args, model, env, file):

    # stage_scheduler = StageScheduler(args)
    # container = Container(args, stage_scheduler=stage_scheduler)
    # env = ProofEnv(args, container, stage_scheduler)
    env.set_file(file)
    env.args.proof_allowed=False

    obs = env.reset()
    t0 = time.time()

    success = False
    step_limit = MIN_STEPS-1
    prove.actions = 0
    while step_limit < MAX_STEPS:
        step_limit += 1
        if time.time() - t0 >= args.evaltime:
            break
        obs = env.reset()
        if prove(model, env, step_limit, obs, t0):
            success = True
            print ("Problem: {}, len {}, depth {}, time: {} sec,\n{}".format(file, prove.actions, step_limit, time.time() - t0, env.real_steps))
            break

    if not success:
        print("Failed to find proof in {} sec, reaching depth {}".format(args.evaltime, step_limit))
    return success

def find_one_proof_nobacktrack(args, model, env, file):
    env.set_file(file)
    env.args.proof_allowed = False
    env.args.max_exploration = None
    env.args.can_replace_proof = False

    success = 0
    prooflen = 0
    for attempts in range(1, 1 + args.evalcount):
        obs = env.reset()
        t0 = time.time()
        status, prooflen = prove_nobacktrack(args, model, env, obs, t0)
        if status == "success":
            print ("Proof found: {}, len {}, time: {} sec,\n{}".format(file, prooflen, time.time() - t0, env.real_steps))
            success = 1
            break
    return success, prooflen, attempts

def eval(args, evaldir, modelfile, model_index):

    model = PPO1.load(modelfile)
    stage_scheduler = StageScheduler(args)
    container = Container(args, stage_scheduler=stage_scheduler)
    env = ProofEnv(args, container, stage_scheduler)

    proofs_found = 0
    proofs_tried = 0
    len_sum = 0.0
    attempts_sum = 0.0
    prove.guidance_time = 0

    dirparts = evaldir.split("/")
    if dirparts[-1] == "":
        dirname = dirparts[-2]
    else:
        dirname = dirparts[-1]
    evalprefix = "eval_{}_{}_{}_{}".format(model_index, dirname, args.evaltype, args.evalcount)

    
    for filename in os.listdir(evaldir):
        if filename.endswith(".p"):
            name = os.path.join(evaldir, filename)
            print("\n\nTrying to find proof for {}".format(name))
            proofs_tried += 1
            success, prooflen, attempts = find_one_proof_nobacktrack(args, model, env, name)
            if success == 1:
                proofs_found += 1
                len_sum += prooflen
                attempts_sum += attempts

        print("Found: {}/{} proofs".format(proofs_found, proofs_tried))

    print("\n\nEVALUATION")
    print("   evaltime: {}".format(args.evaltime))
    print("   evaldir: {}".format(dirname))
    print("   model_index: {}".format(model_index))
    print("   evaltype: {}".format(args.evaltype))
    print("   evalcount: {}".format(args.evalcount))
    print("   FOUND: {}/{}".format(proofs_found, proofs_tried))
    print("   Avg proof length: {}".format(safediv(len_sum, proofs_found)))
    print("   Avg attempts: {}".format(safediv(attempts_sum, proofs_found)))

    # print("Guidance time: {}".format(prove.guidance_time))
    # for t in ["cop_action_time", "cop_backtrack_time", "cop_nos_contras_time", "cop_st_features_time", "cop_contr_features_time", "state_time", "action_time"]:
    #     print("{}: {}".format(t, getattr(env.backend, t)))

def eval_mpi(args, evaldir, modelfile, model_index):

    from mpi4py import MPI as mpi
    rank = mpi.COMM_WORLD.Get_rank()
    all = mpi.COMM_WORLD.Get_size()

    model = PPO1.load(modelfile)
    stage_scheduler = StageScheduler(args)
    container = Container(args, stage_scheduler=stage_scheduler)
    env = ProofEnv(args, container, stage_scheduler)

    dirparts = evaldir.split("/")
    if dirparts[-1] == "":
        dirname = dirparts[-2]
    else:
        dirname = dirparts[-1]

    evalprefix = "eval_{}_{}_{}_{}".format(model_index, dirname, args.evaltype, args.evalcount)
    
    proofs_found = 0
    proofs_tried = 0
    len_sum = 0.0
    attempts_sum = 0.0
    prove.guidance_time = 0

    
    filenames_original = sorted([filename for filename in os.listdir(evaldir) if filename.endswith(".p")])
    def data_gen(filenames,i):
        return filenames[i % len(filenames)]
    chunks = int(len(filenames_original)/all)+1
    filenames_extended = [data_gen(filenames_original,i)
                          for i in range(chunks*all)] # [rank:][::all]
    assert(len(filenames_extended)>0)
    for index in range(chunks):
        chunk = filenames_extended[index*all:(index+1)*all]
        assert(len(chunk)==all)
        name = os.path.join(evaldir, chunk[rank])
        print("\n\nTrying to find proof for {}".format(name))
        success, prooflen, attempts = find_one_proof_nobacktrack(args, model, env, name)
        results = mpi.COMM_WORLD.gather((1, success, prooflen, attempts), root=0)
        if rank == 0:
            # print(results)
            for i in range(len(results)):
                proofs_tried += results[i][0]
                succ = results[i][1]
                if succ == 1:
                    proofs_found += 1
                    len_sum += results[i][2]
                    attempts_sum += results[i][3]
            logger.record_tabular("update_no", proofs_tried)
            logger.record_tabular("{}_proofs_found".format(evalprefix), proofs_found)
            logger.record_tabular("{}_found".format(evalprefix), safediv(proofs_found, proofs_tried))
            logger.record_tabular("{}_avg_prooflen".format(evalprefix), safediv(len_sum, proofs_found))
            logger.record_tabular("{}_avg_attempts".format(evalprefix), safediv(attempts_sum, proofs_found))
            logger.dumpkvs()
            print("Found: {}/{} proofs".format(proofs_found, proofs_tried))


    print("\n\nEVALUATION {}".format(rank))
    print("   evaltime: {}".format(args.evaltime))
    print("   evaldir: {}".format(dirname))
    print("   model_index: {}".format(model_index))
    print("   evaltype: {}".format(args.evaltype))
    print("   evalcount: {}".format(args.evalcount))
    print("   FOUND: {}/{}".format(proofs_found, proofs_tried))
    print("   Avg proof length: {}".format(safediv(len_sum, proofs_found)))
    print("   Avg attempts: {}".format(safediv(attempts_sum, proofs_found)))

    # print("Guidance time: {}".format(prove.guidance_time))
    # for t in ["cop_action_time", "cop_backtrack_time", "cop_nos_contras_time", "cop_st_features_time", "cop_contr_features_time", "state_time", "action_time"]:
    #     print("{}: {}".format(t, getattr(env.backend, t)))



def eval_file(args, evalfile, modelfile, model_index):

    model = PPO1.load(modelfile)
    stage_scheduler = StageScheduler(args)
    container = Container(args, stage_scheduler=stage_scheduler)
    env = ProofEnv(args, container, stage_scheduler)

    prove.guidance_time = 0

    fileparts = evalfile.split("/")
    filename = fileparts[-1]
    evalprefix = "eval_{}_{}_{}_{}".format(model_index, filename, args.evaltype, args.evalcount)

    
    print("\n\nTrying to find proof for {}".format(evalfile))
    success, prooflen, attempts = find_one_proof_nobacktrack(args, model, env, evalfile)

    print("\n\nEVALUATION")
    print("   evaltime: {}".format(args.evaltime))
    print("   evalfile: {}".format(filename))
    print("   model_index: {}".format(model_index))
    print("   evaltype: {}".format(args.evaltype))
    print("   evalcount: {}".format(args.evalcount))
    print("   Success: {}".format(success))
    print("   Proof length: {}".format(prooflen))
    print("   Attempts: {}".format(attempts))
    

def main():
    args, _ = get_configuration(print_diagnostics=True)
    if args.use_previous_action:
        args.state_dim = 5
    else:
        args.state_dim = 4

    PRINT_PROBS = False
    modelfile = "{}/ppo1_fcop_train_0".format(args.outdir)
    # modelfile = "{}/ppo1_fcop_train_nocurr".format(args.outdir)
    args.evaltype="prob"
    args.evalcount=1000

    if False:
        evalfile = "theorems/robinson/random/tiny_one_op/robinson_0p0__0m0.p"
        eval_file(args, evalfile, modelfile, 0)
    else:
        evaldir = "theorems/robinson/left_random/final"
        eval(args, evaldir, modelfile, 0)


if __name__ == '__main__':
    main()
