import matplotlib.pyplot as plt
import matplotlib.animation
import numpy as np
from simulation_run import SimulationRun
from network import Network
from graph_util import GraphUtil as gutil

# Class to visualize consensus, topology and graph data over time.
class Visualizer(object):
    def factory():
        return Visualizer()
    factory = staticmethod(factory)

    def set_data(self, network: Network , run: SimulationRun):
        self.network: Network = network
        self.simulation_run: SimulationRun = run

    def set_node_positions(self, pos):
        self.pos = pos
    #
    # def set_is_spoofed(self, is_spoofed):
    #     self.is_spoofed = is_spoofed

    def get_node_positions(self, time):
        # return np.array([[0.25, 0.75], [0.5, 0.25], [0.75, 0.5]])
        return self.pos

    def get_topology_animation_fn(self, axis):
        N = self.network.num_nodes
        p = self.get_node_positions(time=0)

        topologies = self.simulation_run.get_topologies()
        edges = gutil.edge_list_from_adjacency_matrix(topologies[0, :, :])

        point_style = lambda i: 'bo' if not self.network.is_spoofed(i) else 'ro'
        points = [ axis.plot([p[i, 0]], [p[i, 1]], point_style(i))[0] for i in range(N) ]

        m = []
        for i in range(N):
            for j in range(i+1, N):
                m.append((i, j))

        line_style = lambda edge: 'r-' if self.network.is_spoofed(edge[0]) or self.network.is_spoofed(edge[1]) else 'b-'
        lines = [ axis.plot([],[], line_style((i, j)))[0] for i, j in m ]

        def ani_fn(t):
            p = self.get_node_positions(time=t)
            for i in range(N):
                points[i].set_data([p[i,0]], [p[i,1]])

            idx = 0
            for i, j in m:
                if topologies[t, i, j] > 0:
                    lines[idx].set_data([p[i,0], p[j,0]], [p[i,1],p[j,1]])
                else:
                    lines[idx].set_data([], [])
                idx += 1

        return ani_fn

    def get_value_animation_fn(self, axis):
        N = self.network.num_nodes
        values = self.simulation_run.get_values()

        val_style = lambda i: 'b-' if not self.network.is_spoofed(i) else 'r-'
        values_plots = [ axis.plot([], [], val_style(i), linewidth=0.5)[0] for i in range(N) ]

        true_average_value = self.network.true_average()
        axis.plot([0, self.simulation_run.num_iterations ], [true_average_value, true_average_value], 'g', linewidth=0.7)

        def ani_fn(t):
            for i in range(N):
                values_plots[i].set_data(np.linspace(0,  (t+1), t), values[1:t+1, i])

        return ani_fn

    def get_beta_animation_fn(self, axis):
        N = self.network.num_nodes
        adj = self.network.adjacency_matrix
        betas = self.simulation_run.betas_ij_t

        beta_style = lambda i, j: 'b' if not self.network.is_spoofed(i) and not self.network.is_spoofed(j) else 'r'
        counts = 0
        beta_lines = []
        for i in range(N):
            if self.network.is_spoofed(i): continue
            for j in range(i+1, N):
                if adj[i,j] == 0: continue

                beta_lines.append(axis.plot([], [], beta_style(i,j), linewidth=0.2)[0])
                counts += 1

        def ani_fn(t):
            idx = 0
            for i in range(N):
                if self.network.is_spoofed(i): continue
                for j in range(i+1, N):
                    if adj[i,j] == 0: continue

                    beta_lines[idx].set_data(np.linspace(0, (t+1), t+1), betas[:t+1, i, j])
                    idx += 1

        return ani_fn

    def get_connectivity_animation_fn(self, axis):
        fiedler_line = axis.plot([], [], 'r--')[0]
        legit_fiedler_line = axis.plot([], [], 'b--')[0]

        def ani_fn(t):
            fiedler_line.set_data(np.linspace(0, (t+1), t+1), self.simulation_run.connectivity[:t+1])
            legit_fiedler_line.set_data(np.linspace(0, (t+1), t+1), self.simulation_run.legit_connectivity[:t+1])
        return ani_fn

    # Displays a subplot graphs of the data to visualize
    def show_visualization(self):

        Length = 1000.0 # miliseconds
        T = self.simulation_run.num_iterations

        fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12, 10))
        topology_axis = axes[0, 0]
        values_axis = axes[1, 0]
        betas_axis = axes[1, 1]
        # axes.plot([0, T], [0,0])
        connectivity_axis = axes[0, 1]

        # topology_axis.set_title("Evolution of our Consensus Values and Chosen Network Topologies Over Time")
        topology_axis.set_xlim([-1.2, 1.2])
        topology_axis.set_ylim([-2.0, 2.0])

        # values_axis.set_title("Values")
        values_axis.set_xlim([0, T])
        ymin = min(self.network.node_values[1:]) - 2
        ymax = max(self.network.node_values[1:]) + 2
        print(ymin, ymax)
        values_axis.set_ylim([ymin, ymax])

        # betas_axis.set_title("Beta Values")
        betas_axis.set_ylim([-20, 20])
        betas_axis.set_xlim([0, T])

        conn_ymax = max(np.max(self.simulation_run.connectivity), np.max(self.simulation_run.legit_connectivity)) + 1.0
        conn_ymin = min(np.min(self.simulation_run.connectivity), np.min(self.simulation_run.legit_connectivity)) - 1.0
        # connectivity_axis.set_title("Algebraic Connectivity")
        connectivity_axis.set_ylim([conn_ymin, conn_ymax])
        connectivity_axis.set_xlim([0, T])

        topology_animation = self.get_topology_animation_fn(axis=topology_axis)
        value_animation = self.get_value_animation_fn(axis=values_axis)
        beta_animation = self.get_beta_animation_fn(axis=betas_axis)
        connectivity_animation = self.get_connectivity_animation_fn(axis=connectivity_axis)

        video = True
        if video:
            def animation_function(t):
                topology_animation(t)
                value_animation(t)
                beta_animation(t)
                connectivity_animation(t)
            animation = matplotlib.animation.FuncAnimation(fig=fig, func=animation_function,
                                                           frames=T, interval=Length/T)
        else:
            t = T-2
            topology_animation(t)
            value_animation(t)
            beta_animation(t)
            connectivity_animation(t)

        plt.show()

    def get_visualize_beta_function(self):
        pass

    def show_betas(self):
        self.simulation_run.betas_ij_t[0,:,:]