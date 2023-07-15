import numpy as np

class MabEnvironmentNS:
    def __init__(self, n_arms, n_nodes, T, dim=2, n_phases=1, hf=False):
        # True parameters theta of the model for the activation probabilities
        # self.theta = np.random.dirichlet(np.ones(dim), size=n_phases)
        # vector of correspondence for the features of each individual
        # self.arms_features = [[]]
        # features of each individual (poor/rich and young/old)
        self.nodes_features = np.random.binomial(1, 0.5, size=(n_nodes, dim))
        # probability of each arm (subset of the probability matrix, non-null entries)
        self.p = np.zeros(shape=(n_phases, n_arms))
        # this is the probability matrix for the activation of the edges
        self.prob_matrix = -np.ones(shape=(n_phases, n_nodes, n_nodes))
        # number of active edges
        self.n_arms = n_arms
        # number of nodes on the graph
        self.n_nodes = n_nodes
        # arm -> edges correspondence
        self.arms = []
        self.n_phases = n_phases
        self.T = T
        self.hf = hf
        # we now generate the prob_matrix, edge correspondence features
        self.generate_edge_probs_ns()
        # maybe make it more elegant later
        if n_phases == 1:
            self.prob_matrix = self.prob_matrix[0]


    def generate_edge_probs_ns(self):
        for e in range(self.n_arms):
            i = np.random.randint(self.n_nodes)
            j = np.random.randint(self.n_nodes)
            while i == j or self.prob_matrix[0, i, j] != -1:
                i = np.random.randint(self.n_nodes)
                j = np.random.randint(self.n_nodes)
            self.arms.append(np.array([i, j]))
            for phase in range(self.n_phases):
                self.p[phase, e] = np.random.rand()
                self.prob_matrix[phase, i, j] = self.p[phase, e]
        self.arms = np.array(self.arms)
        self.prob_matrix[self.prob_matrix == -1] = 0

    def round(self, pulled_arm, t):
        if self.hf:
            phase_length = int(np.sqrt(self.T/self.n_phases))
            curr_phase = (t % (phase_length * self.n_phases)) // phase_length
        else:
            curr_phase = int(t // (self.T / self.n_phases))
        return 1 if np.random.random() < self.p[curr_phase, pulled_arm] else 0

    def opt(self, t):
        if self.hf:
            phase_length = int(np.sqrt(self.T / self.n_phases))
            curr_phase = (t % (phase_length * self.n_phases)) // phase_length
        else:
            curr_phase = int(t // (self.T / self.n_phases))
        return np.max(self.p[curr_phase, :])

    def simulate_episode(self, set_seeds):
        n_max_steps = self.n_nodes
        prob_matrix = self.prob_matrix.copy()
        initial_active_nodes = set_seeds
        history = np.array([initial_active_nodes])
        active_nodes = initial_active_nodes
        newly_active_nodes = active_nodes
        t = 0
        while t < n_max_steps and np.sum(newly_active_nodes) > 0:
            # p is a vector (n_nodes x 1) that contains in the i-th entrance  the probability
            # that node i is going to be activated.
            p = (prob_matrix.T * active_nodes).T
            activated_edges = p > np.random.rand(p.shape[0], p.shape[1])
            # 3 possible cases:
            # 1) p == 0 and activated_edge == 0: a node that has already been activated -> we keep the prob matrix untouched
            # 2) p != 0 and activated_edge == 1: a node that has been activated in this round -> we keep the prob matrix untouched
            # 3) p != 0 and activated edge == 0: a node that hasn't been activated in this round (but it could've been)
            #                                    -> we set this edge to 0 on the probability matrix
            prob_matrix = prob_matrix * ((p != 0) == activated_edges)
            # we sum by row activated_edges, so that if at least one edge has activated the j-th node we are going to try
            # to activate the j-th node. Then we activate the j-th node if the latter is not in the active_nodes
            newly_active_nodes = (np.sum(activated_edges, axis=0) > 0) * (1 - active_nodes)
            active_nodes = np.array(active_nodes + newly_active_nodes)
            history = np.concatenate((history, [newly_active_nodes]), axis=0)
            t += 1
        return history

    def customer_class(self):
        cc = np.array([])
        for f in self.nodes_features:
            # C1 = (0,0) or (0,1) this is the class of low-income individuals
            if f[0] == 0:
                cc = np.append(cc, 0)
            # C2 = (1,0) this is the class of rich, young people
            if f[0] == 1 and f[1] == 0:
                cc = np.append(cc, 1)
            # C3 = (1,1) this is the class of rich, old people
            if f[0] == 1 and f[1] == 1:
                cc = np.append(cc, 2)
        return cc


