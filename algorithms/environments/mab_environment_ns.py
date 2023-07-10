import numpy as np

class MabEnvironmentNS:
    def __init__(self, n_arms, n_nodes, T, n_phases, hf=False):
        # True parameters theta of the model for the activation probabilities
        # self.theta = np.random.dirichlet(np.ones(dim), size=n_phases)
        # vector of correspondence for the features of each individual
        # self.arms_features = [[]]
        # features of each individual (poor/rich and young/old)
        # self.nodes_features = np.random.binomial(1, 0.5, size=(n_phases, n_nodes, dim))
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

