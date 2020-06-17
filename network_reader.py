from network import Network
from simulation_run import SimulationRun
from alpha_generator import AlphaGenerator
from switching_strategy import SwitchingFunctionStrategy, NoSwitchingStrategy
from visualizer import Visualizer
import matplotlib.pyplot as plt
import numpy as np

class CSVNetworkReader(object):

    def __init__(self, filename):
        self.f = open(file=filename, mode='r')
        if self.f.mode != 'r':
            raise Exception(self.f.errors)

    def read_network(self, matlab=False) -> Network:
        num_legit, num_spoof = list(map(int, self.f.readline().split(",")))
        legit_values = list(map(float, self.f.readline().split(",")))
        assert(len(legit_values) == num_legit)
        spoofed_value = float(self.f.readline())

        network = Network(
            num_nodes=num_legit+num_spoof,
            num_spoofed=num_spoof
        )

        n_val = lambda i: legit_values[i] if i < num_legit else spoofed_value
        network.node_values = np.array([n_val(i) for i in range(num_legit + num_spoof)])

        num_edges = int(self.f.readline())
        for i in range(num_edges):
            i, j = list(map(int, self.f.readline().split(',')))
            if matlab:
                i -= 1
                j -= 1
            network.adjacency_matrix[i,j] = 1
            network.adjacency_matrix[j,i] = 1

        print(legit_values, spoofed_value)
        return network

    def close(self):
        self.f.close()


if __name__ == "__main__":
    # np.random.seed(42)
    filename = "case1_data.csv"
    reader = CSVNetworkReader(filename)
    network = reader.read_network(matlab=True)
    network.print()
    reader.close()

    visualizer = Visualizer.factory()


    # is_switching flag when true turn off links when beta values drop below a threshold, if false
    # then do not change the topology with the beta values
    is_switching = True

    # resilience flag true when resilience is used in weights, if false then use standard consensus
    # weights such as 1.0 / N
    resilience = True

    # If changing_spoofers is true override the initial spoof value, given in the csv data, with a
    # constantly changing value with periodic values (such as sine or cosine)
    changing_spoofers = False

    strategy = None # there is no strategy
    if is_switching:
        strategy = SwitchingFunctionStrategy(
            network=network,
            topology=np.array([
                [1 if i != j else 0 for j in range(network.num_nodes)]
                for i in range(network.num_nodes)
            ]),
            fn= lambda betas: betas > -0.2
        )
    else:
        strategy = NoSwitchingStrategy(network=network)

    # dspace = np.linspace(0.1, 0.35, 400)
    # # stdspace = np.linspace(0.01, 0.3, 20)
    #
    # num_legit = network.num_nodes-network.num_spoofed
    #
    # disagreement_vals = []
    # for i in range(400):
    #     alpha_generator = AlphaGenerator(network=network, stdev=0.33, delta=dspace[i])
    #     sim_run = SimulationRun(
    #         network=network,
    #         strategy=strategy,
    #         alpha_gen=alpha_generator,
    #         resilience=resilience,
    #         sinewave_spoof=changing_spoofers
    #     )
    #     sim_run.run_simulation()
    #     values = sim_run.get_values()
    #     true_avg = network.true_average()
    #     disagreement = [ np.sqrt(np.sum(np.square(values[t, :num_legit] - true_avg)) / num_legit) for t in range(values.shape[0]) ]
    #     disagreement_vals.append(disagreement)
    #     if i % 50 == 0: print(i)
    #
    # disagreement_vals = np.array(disagreement_vals)
    #
    # means = np.array([ np.average(disagreement_vals[:, i]) for i in range(disagreement_vals.shape[1]) ])
    # variances = np.array([ np.var(disagreement_vals[:, i]) for i in range(disagreement_vals.shape[1]) ])
    #
    #
    # oscillating = "" if not changing_spoofers else "_osc"
    # case = 2 if not is_switching and not resilience else ( 1 if is_switching else 0 )
    # results_file = "{0}.results_sw{1}{2}.csv".format(filename, case, oscillating)
    # f_res = open(results_file, 'w')
    # if f_res.mode != 'w':
    #     raise Exception("Can't write results")
    #
    # f_res.write("{0}\n".format(means.shape[0]))
    # f_res.write("{0}\n".format(variances.shape[0]))
    # for i in range(means.shape[0]):
    #     f_res.write("{0}, ".format(means[i]))
    # f_res.write("\n")
    # for i in range(variances.shape[0]):
    #     f_res.write("{0}, ".format(variances[i]))
    # f_res.write("\n")
    # f_res.close()
    # print("done: results written to ", results_file)
    #
    # x = np.linspace(0, 99, 100)
    # plt.plot(x, means)
    # plt.fill_between(x, means - variances, means + variances, alpha=0.1)
    # # plt.errorbar(x, means, variances)
    # plt.show()


    alpha_generator = AlphaGenerator(network=network, stdev=0.33, delta=0.35)
    sim_run = SimulationRun(
        network=network,
        strategy=strategy,
        alpha_gen=alpha_generator,
        resilience=resilience,
        sinewave_spoof=changing_spoofers
    )
    sim_run.run_simulation()
    # print(sim_run.values_t)
    # ploy them in a circle
    visualizer.set_node_positions(pos=np.array([
        [np.sin(theta), np.cos(theta)]
        for theta in np.linspace(
            start=0,
            stop=2*np.pi,
            num=network.num_nodes + 1)[:-1]
    ]))

    visualizer.set_data(network=network, run=sim_run)
    visualizer.show_visualization()