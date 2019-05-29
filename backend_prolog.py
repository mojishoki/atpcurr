from pyswip import Prolog

prolog = Prolog()

class Backend(object):
    def __init__(self, indexer, n_dim, backtrack=False, verbose=True, print_proof=True):
        self.prover = prolog.consult("leancop_rl.pl")
        self.indexer = indexer
        self.settings = ["conj","nodef", "n_dim({})".format(n_dim)]
        self.verbose = verbose
        if verbose:
            self.settings.append('verbose')
        if print_proof:
            self.settings.append('print_proof')
        if backtrack:
            self.settings.append('backtrack')
        
    def start(self, file, pathLim=0):
        self.file = file
        settings = ','.join(self.settings)
        if pathLim > 0:
            settings += ',pathLim({})'.format(pathLim)

        query = "embed_init(\"{}\",[{}],[Goal,Path,Lem,Todos],Actions,Result).".format(self.file, settings)
        if True: #self.verbose:
            print("INIT: \n",query)
        result = list(prolog.query(query))
        result = result[0]
        goal = result['Goal']
        path = result['Path']
        lem = result['Lem']
        todos = result['Todos']
        action_list = result['Actions']
        done = result['Result']
        return self.process(action_list, goal, path, lem, todos, done)
    def restart(self, pathLim=0):
        return self.start(self.file, pathLim)
    def step(self, action_index):
        query = "embed_step({},[Goal,Path,Lem,Todos],Actions,Result).".format(action_index)
        if True: # self.verbose:
            print("STEP: \n",query)
        result = list(prolog.query(query))
        result = result[0]
        goal = result['Goal']
        path = result['Path']
        lem = result['Lem']
        todos = result['Todos']
        action_list = result['Actions']
        done = result['Result']
        return self.process(action_list, goal, path, lem, todos, done)
    def print_state(self):
        pass # TODO

    def process(self, action_list, goal, path, lem, todos, done):
        
        embedded_actions = [self.indexer.store_action(action, pairs=True) for action in action_list]
        embedded_action_ids = [a[0] for a in embedded_actions]
        embedded_action_list = [a[1] for a in embedded_actions]
        if done == 0:
            state_id, state = self.indexer.store_state(goal, path, lem, todos, pairs=True)
        elif done == 1:
            state_id, state = "success", 0
        elif done == -1:
            state_id, state = "failure", 0
        if self.verbose:
            print("action list: ", action_list)
            print("goal: ", goal)
            print("path: ", path)
            print("lem:  ", lem)
            print("todos:", todos)
            print("state: ", state_id)
        return state, state_id, embedded_action_list, embedded_action_ids, done
