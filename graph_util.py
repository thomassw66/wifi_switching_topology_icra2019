import numpy as np
import heapq

class GraphUtil(object):

    @staticmethod
    def degree_matrix(adjacency_matrix: np.array):
        return np.identity(adjacency_matrix.shape[0], dtype=np.int) * np.sum(adjacency_matrix, axis=0)

    @staticmethod
    def laplacian(adjacency_matrix: np.array):
        """ Calculates the Laplacian of the graph given by the adjacency matrix defined as the
            degree matrix less the adjacency matrix """
        return GraphUtil.degree_matrix(adjacency_matrix) - adjacency_matrix

    @staticmethod
    def adjacency_matrix_from_edge_list(edge_list: np.array, num_elements: int):
        """ Converts a graph given as an edge list into an equivalent adjacency matrix in O(|E|)
            where |E| is the cardinality of the set of edges """
        adj = np.zeros(shape=(num_elements, num_elements))
        for edge in edge_list:
            i = edge[0]
            j = edge[1]
            adj[i,j] = 1
            adj[j,i] = 1
        return adj

    @staticmethod
    def edge_list_from_adjacency_matrix(adjacency_matrix: np.array):
        """ Converts a graph given as an adjacency matrix into an equivalent edge list in O(N^2) """
        N = adjacency_matrix.shape[0]
        edge_list = []
        for i in range(N):
            for j in range(i+1, N):
                if adjacency_matrix[i,j] > 0:
                    edge_list.append((i, j))
        return edge_list

    @staticmethod
    def fiedler_val(L):
        """ Calculates the Algebraic connectivity (of Fiedler Value equivalently) of L """
        eigens = np.linalg.eigh(L)[0]
        # print("Eigen")
        # print(eigens)
        return heapq.nsmallest(2, eigens)[-1]