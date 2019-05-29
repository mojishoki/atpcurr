# Towards Finding Longer Proofs

We present a reinforcement learning based guidance system for
automated theorem proving geared towards Finding Longer Proofs (FLoP).
FLoP focuses on generalizing from short proofs to longer ones of
similar structure. To achieve that, FLoP uses state-of-the-art RL
approaches that were previously not applied in theorem proving. In
particular, we show that curriculum learning significantly outperforms
previous learning-based proof guidance on a synthetic dataset of
increasingly difficult arithmetic problems.  The proof engine used by
FLoP is based on a connection calculus and specifically on leanCoP and
its OCaml implementation introduced in FEMaleCoP.

The dataset and the training algorithm are described in details in the
paper. Supplementary materials including screencasts with gameplays performed
in our environments are available at the project webpage
http://bit.ly/site_atpcurr

**Datasets**
The dataset that we use in our experiments is based on Robinson Arithmetic and consists of 3
stages of increasing complexity. Problems can be found at:

- [Stage 1: theorems/robinson/simple/final](theorems/robinson/simple/final)
- [Stage 2: theorems/robinson/left_random/final](theorems/robinson/left_random/final)
- [Stage 3: theorems/robinson/random/final2](theorems/robinson/random/final2)

**Data generation**

We are using simple, synthetic datasets, which makes it very easy to
generate different variants. The codebase includes a data generator
which can be used e.g.:

```
python generators/gen_random.py --preamble_file generators/peano_fof.p
--count 300 --type pairs --first_limit 10 --op_count 3 --ops
"plus|10,mul|10" --output_dir /theorems/robinson/random/final2
```

This code generates problems in Robinson Arithmetic such that the
conjecture is a ground arithmetic equation with 3 operators on both
sides (using only addition and multiplication), with operators up to 10.


**Experiments**

Experiment parameters are described in configuration files. Examples can
be found in directory [ini](ini).

**Usage**

Running the code is as simple as this:

```
python train_ppo.py --ex {configuration file}
```

e.x.:

```
python train_ppo.py --ex ini/experiment_robinson_noproof_simple_MPI.py
```

An experiment consists of training a model on the dataset specified by
the configuration file and then running evaluation on the evaluation
on the test set.

**Included software**

This distribution consists of:

- All our arithmetic datasets
- A data generator
- Configuration files used in the final experiments
- The complete guidance system built using the Proximal Policy Optimization (PPO) implementation of Stable Baselines https://github.com/hill-a/stable-baselines/tree/master/stable_baselines. 

Two components of the software are excluded:

- Binary with the OCaml engine: The binary cannot be publicly released at this time and is distributed on request
- The experiment runner: The runner is directly linked to our hardware infrastructure and would be useless elsewhere 
