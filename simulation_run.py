import numpy as np
from graph_util import GraphUtil as gutil
import scipy
import network
import switching_strategy


class SimulationRun(object):

    num_iterations: int
    strategy: switching_strategy.SwitchingStrategy
    network: network.Network

    def __init__(self, network: network.Network, strategy: switching_strategy.SwitchingStrategy,
                 alpha_gen, resilience=True, simulations_steps=100, sinewave_spoof=False):
        self.num_iterations = simulations_steps
        self.has_run = False
        self.network = network
        self.strategy = strategy
        self.alpha_gen = alpha_gen
        self.resilience = resilience
        self.sinewave_spoof = sinewave_spoof

    def run_simulation(self):
        T = self.num_iterations
        N = self.network.num_nodes
        Ns = self.network.num_spoofed
        Nl = N - Ns
        self.values_t = np.zeros(shape=(T,N))
        self.values_t[0, :] = self.network.node_values.copy()
        self.topologies_t = np.zeros(shape=(T,N,N))
        self.betas_ij_t = np.zeros(shape=(T,N,N))
        self.alphas_ij_t = np.zeros(shape=(T,N,N))
        self.connectivity = np.zeros(shape=(T))
        self.legit_connectivity = np.zeros(shape=(T))
        for t in range(0, T-1):
            self.alphas_ij_t[t, :, :] = self.alpha_gen.get_alpha_ij(time=t)
            self.betas_ij_t[t, :, :] += self.alphas_ij_t[t, :, :]
            self.betas_ij_t[t+1, :, :] = self.betas_ij_t[t, :, :]
            is_switching = False
            if type(self.strategy) is switching_strategy.SwitchingFunctionStrategy:
                is_switching = True
                self.topologies_t[t, :, :] = self.strategy.get_topology(betas=self.betas_ij_t[t, :, :])
            else:
                is_switching = False
                self.topologies_t[t, :, :] = self.strategy.get_topology(time=t)
            self.connectivity[t] = gutil.fiedler_val(gutil.laplacian(self.topologies_t[t, :, :]))
            self.legit_connectivity[t] = gutil.fiedler_val(gutil.laplacian(self.topologies_t[t, :Nl, :Nl]))
            for i in range(N):
                if self.network.is_spoofed(i):
                    if self.sinewave_spoof:
                        shift = 1
                        period = 1
                        y = 5.31
                        mag = 2.5
                        self.values_t[t+1, i] = mag * np.sin((t-shift)/period) + y
                    else:
                        self.values_t[t+1, i] = self.values_t[t, i]
                    continue

                Ni = np.count_nonzero(self.topologies_t[t,i, :]) + 1
                divisor = N if is_switching else Nl
                diff = lambda i, j: self.values_t[t, j] - self.values_t[t, i]
                weight = lambda i, j: (0.5 * np.exp(self.betas_ij_t[t, i, j])) \
                    if self.betas_ij_t[t, i, j] < 0 \
                    else (1.0 - np.exp(- self.betas_ij_t[t, i, j] / 2.0))
                if self.resilience:
                    self.values_t[t+1, i] = self.values_t[t, i] + np.sum(np.array([
                        1.0 / divisor * weight(i, j) * diff(i, j) for j
                        in range(N)
                        if self.topologies_t[t, i, j] == 1
                            and i != j
                    ]))
                else:
                    self.values_t[t+1, i] = self.values_t[t, i] + np.sum(np.array([
                        1.0 / self.network.num_nodes * diff(i, j) for j
                        in range(N)
                        if self.topologies_t[t, i, j] == 1
                            and i != j
                    ]))
        self.has_run = True


    def get_values(self):
        if not self.has_run: raise Exception("Simulation values accessed but simulation not yet run")
        return self.values_t

    def get_topologies(self):
        if not self.has_run: raise Exception("Simulation topologies accessed but simulation not yet run")
        return self.topologies_t