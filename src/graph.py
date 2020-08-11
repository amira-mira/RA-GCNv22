import numpy as np


class Graph():
    def __init__(self, max_hop=1, dilation=1):
        self.max_hop = max_hop
        self.dilation = dilation

        # get edges
        self.num_node, self.edge, self.center = self._get_edge()
        
        # get adjacency matrix
        self.hop_dis = self._get_hop_distance()
        
        # normalization
        self.A = self._get_adjacency()

    def __str__(self):
        return self.A

    def _get_edge(self):
        num_node = 25
        neighbor_1base = [(1, 2), (2, 21), (3, 21), (4, 3), (5, 21),
                          (6, 5), (7, 6), (8, 7), (9, 21), (10, 9),
                          (11, 10), (12, 11), (13, 1), (14, 13), (15, 14),
                          (16, 15), (17, 1), (18, 17), (19, 18), (20, 19),
                          (22, 23), (23, 8), (24, 25), (25, 12)]
        self_link = [(i, i) for i in range(num_node)]
        neighbor_link = [(i - 1, j - 1) for (i, j) in neighbor_1base]
        edge = self_link + neighbor_link
        center = 21 - 1
        return (num_node, edge, center)

    def _get_hop_distance(self):
        A = np.zeros((self.num_node, self.num_node))
        for i, j in self.edge:
            A[j, i] = 1
            A[i, j] = 1
        hop_dis = np.zeros((self.num_node, self.num_node)) + np.inf
        transfer_mat = [np.linalg.matrix_power(A, d) for d in range(self.max_hop + 1)]
        arrive_mat = (np.stack(transfer_mat) > 0)
        for d in range(self.max_hop, -1, -1):
            hop_dis[arrive_mat[d]] = d
        return hop_dis

    def _get_adjacency(self):
        valid_hop = range(0, self.max_hop + 1, self.dilation)
        adjacency = np.zeros((self.num_node, self.num_node))
        for hop in valid_hop:
            adjacency[self.hop_dis == hop] = 1
        normalize_adjacency = self._normalize_digraph(adjacency)
        A = np.zeros((len(valid_hop), self.num_node, self.num_node))
        for i, hop in enumerate(valid_hop):
            A[i][self.hop_dis == hop] = normalize_adjacency[self.hop_dis == hop]
        return A

    def _normalize_digraph(self, A):
        Dl = np.sum(A, 0)
        num_node = A.shape[0]
        Dn = np.zeros((num_node, num_node))
        for i in range(num_node):
            if Dl[i] > 0:
                Dn[i, i] = Dl[i]**(-1)
        AD = np.dot(A, Dn)
        return AD

