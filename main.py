import numpy as np
from scipy import linalg
from visualizer import Visualizer
from network import Network
from simulation_run import SimulationRun
from graph_util import GraphUtil as gutil
import switching_strategy
from alpha_generator import AlphaGenerator

# network = Network(num_nodes=6)
# for i in range(network.num_nodes):
#     network.set_value(i, val=np.random.normal(loc=0.0, scale=3))
# # network.node_values = np.array([-0.3865356, 0.84706245, 0.84706245, 0.84706245, 0.84706284, 0.84706139])
# print(np.average(network.node_values))
# input("Press enter:")
# network.set_weight(0, 1, weight=0.3)
# network.set_weight(1, 2, weight=0.3)
# network.set_weight(3, 4, weight=0.2)
# network.set_weight(3, 5, weight=0.1)
# network.set_weight(2, 3, weight=0.3)
# network.print()

NUM_NODES = 8
NUM_SPOOF = 2

is_spoofed = [ False if i < NUM_NODES - NUM_SPOOF else True for i in range(NUM_NODES) ]

# np.random.seed(4243)
# np.random.seed(4246)
network = Network.random_network_factory(num_nodes=NUM_NODES,
                                         value_generator=(lambda x: np.random.uniform(low=0, high=10)),
                                         average_degree=7,
                                         num_spoofed=NUM_SPOOF, spoofed_val=10)

# top = gutil.adjacency_matrix_from_edge_list(num_elements=NUM_NODES, edge_list=[[0,1], [0,4], [0,2], [3, 4], [3,5], [6,7], [7,8], [6,8], [5,6], [0,3], [1,8],[2,7], [0,5]])
# tp_list = [
    # top
# ]
# tp_list = np.array(tp_list)

# network.adjacency_matrix = top
network.print()
true_ave = np.sum(network.node_values[: NUM_NODES - NUM_SPOOF ]) / (network.num_nodes - NUM_SPOOF)
print(true_ave)
L = gutil.laplacian(network.adjacency_matrix)
print(L)
print(gutil.fiedler_val(L))
print(is_spoofed)
input("press enter")



# strat = switching_strategy.RandomSwitchingStrategy(network=network, p=0.5)
# strat = switching_strategy.TimedSwitchingStrategy(network=network, topology_list=tp_list)
# strat = switching_strategy.SwitchingFunctionStrategy(network=network, topology=network.adjacency_matrix, fn=(lambda betas: betas > 1))
strat = switching_strategy.NoSwitchingStrategy(network=network)
# generate a random network with values products by this funciton f that has an average degree of d

sim_run = SimulationRun(network=network, strategy=strat,
                        alpha_gen=AlphaGenerator(network=network, delta=0.35, stdev=0.25), simulations_steps=150, resilience=False, sinewave_spoof=True)
sim_run.run_simulation()

print(sim_run.connectivity)
print(sim_run.legit_connectivity)

vis = Visualizer.factory()
vis.set_node_positions(pos=np.array([[np.sin(theta), np.cos(theta)] for theta in np.linspace(start=0, stop=2*np.pi, num=NUM_NODES+1)[:-1]]))
# vis.set_is_spoofed(is_spoofed=is_spoofed)
print(vis.get_node_positions(time=0))
vis.set_data(network=network, run=sim_run)

params = {'topology': True, 'values': True, 'betas': True}

vis.show_visualization()
