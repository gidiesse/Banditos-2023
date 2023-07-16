import numpy as np

"""
This is the stationary linear environment used for generating the probabilities
of activation for each edge of the graph. 
The environment will randomly generate a vector of parameters theta and a vector
of features x, with which we can compute the probabilities of activation for the
edges of the graph (arms of the bandit) according to the formula p = x'theta. 
Our goal will be to learn the activation probabilities of the edges.
Remember that the feature vector x is known, something we can observe, e.g.
it could be a vector of 2 entries with:
x[0] = 0 if the candidate is poor, x[0] = 1 if the candidate is rich
x[1] = 0 if the candidate is young, x[1] = 1 if the candidate is old 
Instead, theta is the part of the model which is unknown and that we have to learn.
"""


class LinearMabEnvironment:
    # n_arms is the number of active edges (50 in our case)
    # dim is the dimension of our feature vector (2 in our case)
    def __init__(self, n_arms, dim, n_nodes):
        # True parameters theta of the model for the activation probabilities
        self.theta = np.random.dirichlet(np.ones(dim), size=1)
        # vector of correspondence for the features of each individual
        self.arms_features = []
        # features of each individual (poor/rich and young/old)
        self.nodes_features = np.random.binomial(1, 0.5, size=(n_nodes, dim))
        # probability of each arm (subset of the probability matrix, non-null entries)
        self.p = np.zeros(n_arms)
        # this is the probability matrix for the activation of the edges
        self.prob_matrix = -np.ones(shape=(n_nodes, n_nodes))
        # number of active edges
        self.n_arms = n_arms
        # number of nodes on the graph
        self.n_nodes = n_nodes
        # arm -> edges correspondence
        self.arms = []
        # we now generate the prob_matrix, edge correspondence features
        self.generate_edge_probs()

    def round(self, pulled_arm):
        return 1 if np.random.random() < self.p[pulled_arm] else 0

    def opt(self):
        return np.max(self.p)

    def generate_edge_probs(self):
        features = []
        for e in range(self.n_arms):
            i = np.random.randint(self.n_nodes)
            j = np.random.randint(self.n_nodes)
            while i == j or self.prob_matrix[i, j] != -1:
                i = np.random.randint(self.n_nodes)
                j = np.random.randint(self.n_nodes)
            arm_feature = (self.nodes_features[i, ] == self.nodes_features[j, ]).astype(int)
            features.append(arm_feature)
            self.arms.append(np.array([i, j]))
            self.p[e] = np.dot(self.theta, arm_feature)
            self.prob_matrix[i, j] = self.p[e]
        self.arms_features = np.array(features)
        self.arms = np.array(self.arms)
        self.prob_matrix[self.prob_matrix == -1] = 0


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

