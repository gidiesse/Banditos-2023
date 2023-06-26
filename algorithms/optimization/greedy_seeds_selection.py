import numpy as np

"""
Greedy algorithm for selecting the best subset of seeds in an influence maximisation problem 
given the probability matrix for the activation of the edges 
"""
class GreedySeedsSelection:
    def __init__(self, prob_matrix, n_iterations, n_max_steps):
        self.seeds_set = np.array([])
        self.p_matrix = prob_matrix
        self.n_nodes = prob_matrix.shape[0]
        self.n_iterations = n_iterations
        self.nodes = np.array(range(self.n_nodes))
        self.n_max_steps = n_max_steps

    def simulate_episode(self, init_prob_matrix, initial_active_nodes):
        # remember that initial active nodes must be a np.array
        prob_matrix = init_prob_matrix.copy()
        active_nodes = initial_active_nodes
        newly_active_nodes = active_nodes
        t = 0
        while t < self.n_max_steps and np.sum(newly_active_nodes) > 0:
            # p is a vector (n_nodes x 1) that contains in the i-th entrance  the probability
            # that node i is going to be activated.
            p = (prob_matrix.T * active_nodes).T
            activated_edges = p > np.random.rand(p.shape[0], p.shape[1])
            # 3 possible cases:
            # 1) p == 0 and activated_edge == 0: a node that has already been activated -> we keep the prob matrix untouched
            # 2) p != 0 and activated_edge == 1: a node that has been activated in this round -> we keep the prob matrix untouched
            # 3) p != 0 and activated edge == 1: a node that hasn't been activated in this round (but it could've been)
            #                                    -> we set this edge to 0 on the probability matrix
            prob_matrix = prob_matrix * ((p != 0) == activated_edges)
            # we sum by row activated_edges, so that if at least one edge has activated the j-th node we are going to try
            # to activate the j-th node. Then we activate the j-th node if the latter is not in the active_nodes
            newly_active_nodes = (np.sum(activated_edges, axis=0) > 0) * (1 - active_nodes)
            active_nodes = np.array(active_nodes + newly_active_nodes)
            t += 1
        return np.sum(active_nodes)

    def expected_activations(self, initial_active_nodes):
        count = 0
        for e in range(self.n_iterations):
            count += self.simulate_episode(self.p_matrix, initial_active_nodes)
        return count / self.n_iterations

    def select_seeds(self, k):
        available_nodes = self.nodes
        seeds_set = np.zeros(self.n_nodes)
        for i in range(k):
            max_activations = 0
            best_seed = 0
            for j in available_nodes:
                initial_active_nodes = seeds_set.copy()
                initial_active_nodes[j] += 1
                expected_activations = self.expected_activations(initial_active_nodes)
                if expected_activations > max_activations:
                    max_activations = expected_activations
                    best_seed = j
            seeds_set[best_seed] += 1
            available_nodes = np.setdiff1d(available_nodes, np.array([best_seed]))
        return seeds_set



