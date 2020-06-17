import numpy as np

# A network represents a simple undirected graphs which can be used to model the positions
class Network(object):

    def __init__(self, num_nodes, num_spoofed):
        self.num_nodes = num_nodes
        self.node_values = np.zeros(shape=(num_nodes,), dtype=float)
        self.num_spoofed = num_spoofed
        self.adjacency_matrix = np.zeros(shape=(num_nodes, num_nodes), dtype=np.int)

    @staticmethod
    def random_network_factory(num_nodes=10, value_generator=(lambda x: x), average_degree=4, num_spoofed=0, spoofed_val=10):
        prob = float(average_degree) / num_nodes
        network = Network(num_nodes=num_nodes, num_spoofed=num_spoofed)
        value = lambda i: value_generator(i) if not network.is_spoofed(i) else spoofed_val
        network.node_values = np.array([ value(i) for i in range(num_nodes) ], dtype=np.float)
        for i in range(num_nodes):
            for j in range(i+1, num_nodes):
                if np.random.choice(a=[0, 1], p=[1 - prob, prob]):
                    network.add_edge(i, j)
        return network

    def is_spoofed(self, i):
        return i >= self.num_nodes - self.num_spoofed

    def add_edge(self, i: int, j: int):
        # if i == j: return # not allowed to have weights on self
        self.adjacency_matrix[i][j] = 1
        self.adjacency_matrix[j][i] = 1

    def set_value(self, i: int, val: np.float):
        self.node_values[i] = val

    def true_average(self):
        num_legit = self.num_nodes - self.num_spoofed
        if num_legit <= 0:
            return 0

        return sum([
            self.node_values[i] for i
            in range(self.num_nodes)
            if not self.is_spoofed(i)
        ]) / num_legit

    def print(self):
        print("Values:")
        print(self.node_values, sep=" ", end="\n")
        print("Edge Weights:")
        print(self.adjacency_matrix)

    def get_degree(self):
        return np.identity(self.num_nodes) * np.sum(self.adjacency_matrix, axis=0)

    def laplacian(self):
        return self.get_degree() - self.adjacency_matrix