import numpy as np

class AlphaGenerator(object):
    def __init__(self, network, stdev=1.0, delta=0.25):
        # self.N = self.network.num_nodes
        self.stdev = stdev
        self.network = network
        self.delta = delta

    def get_alpha_ij(self, time):
        N = self.network.num_nodes
        alpha_ij = np.zeros(shape=(N,N))
        for i in range(N):
            if not self.network.is_spoofed(i): # only generate alphas for real nodes
                for j in range(i+1, N):
                    if self.network.is_spoofed(j):
                        alpha_ij[i,j] = np.clip(np.random.normal(loc=(self.delta - 0.5), scale=self.stdev), a_min=-0.5, a_max=0.5)
                        alpha_ij[j,i] = alpha_ij[i, j]
                    else:
                        alpha_ij[i,j] = np.clip(np.random.normal(loc=(1.0 - self.delta - 0.5), scale=self.stdev), a_min=-0.5, a_max=0.5)
                        alpha_ij[j,i] = alpha_ij[i,j]
                        # alpha_ij[j,i] = np.random.normal(loc=0.25, scale=self.stdev)
        return alpha_ij