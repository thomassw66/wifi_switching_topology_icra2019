import numpy as np
import network

class SwitchingStrategy(object):
    def get_topology(self, time: int) -> np.ndarray:
        pass

class RandomSwitchingStrategy(SwitchingStrategy):
    """
    A switching strategy that, given an initial network, generates topologies by switching edges
    off with some probability
    """
    def __init__(self, network: network.Network, p: np.float = 0.5):
        """

        :type p: object
        """
        self.network = network
        self.p = p

    def get_topology(self, time: int) -> np.ndarray:
        A = self.network.adjacency_matrix.copy()
        for i in range(self.network.num_nodes):
            for j in range(i + 1, self.network.num_nodes):
                if A[i, j] > 0:
                    A[i, j] = np.random.choice(a=[0, 1], p=[self.p, 1.0 - self.p])
                    A[j, i] = A[i, j]
        return A

class TimedSwitchingStrategy(SwitchingStrategy):
    def __init__(self, network: network.Network, topology_list=[], total_time=100):
        self.network = network
        self.num_switches = len(topology_list)
        self.topology_list = topology_list
        self.switching_freq = total_time / self.num_switches

    def get_topology(self, time: int) -> np.ndarray:
        index = int (time / self.switching_freq)
        return self.topology_list[index].copy()

class ModuloSwitchingStrategy(SwitchingStrategy):
    def __init__(self, network: network.Network, topology_list=[]):
        self.network = network
        self.topology_list = topology_list

    def get_topology(self, time: int) -> np.ndarray:
        return self.topology_list[time % len(self.topology_list)].copy()

class SwitchingFunctionStrategy(SwitchingStrategy):
    def __init__(self, network, fn, topology=None):
        self.network = network
        self.topology = topology if topology is not None else network.adjacency_matrix
        self.fn = fn

    def get_topology(self, betas: np.ndarray) -> np.ndarray:
        return np.multiply(self.fn(betas), self.topology)
        # return self.topology
        # return self.network.adjacency_matrix

class NoSwitchingStrategy(SwitchingStrategy):
    def __init__(self, network):
        self.network = network

    def get_topology(self, betas):
        return self.network.adjacency_matrix

    def get_topology(self, time: int):
        return self.network.adjacency_matrix