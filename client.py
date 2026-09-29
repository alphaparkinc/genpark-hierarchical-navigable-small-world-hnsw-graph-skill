"""HNSW Graph Vector Index.
100% Python Standard Library.
"""

import math
import heapq
from collections import defaultdict

def euclidean_dist(v1, v2):
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(v1, v2)))

class HNSWGraph:
    """Pure Python Hierarchical Navigable Small World vector index."""
    def __init__(self, m=4, ef_search=8):
        self.m = m
        self.ef_search = ef_search
        self.nodes = {}
        self.layers = [defaultdict(list)]
        self.enter_node = None

    def insert(self, node_id, vector):
        self.nodes[node_id] = vector
        if self.enter_node is None:
            self.enter_node = node_id
            return

        current = self.enter_node
        dists = [(euclidean_dist(vector, self.nodes[current]), current)]
        visited = {current}

        while dists:
            curr_dist, curr_node = heapq.heappop(dists)
            improved = False
            for neighbor in self.layers[0][curr_node]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    d = euclidean_dist(vector, self.nodes[neighbor])
                    if d < curr_dist:
                        heapq.heappush(dists, (d, neighbor))
                        improved = True
            if not improved:
                break

        closest = sorted(visited, key=lambda nid: euclidean_dist(vector, self.nodes[nid]))[:self.m]
        for n in closest:
            self.layers[0][node_id].append(n)
            self.layers[0][n].append(node_id)

    def search(self, query_vec, top_k=3):
        if not self.nodes:
            return []
        visited = set()
        candidates = []
        start = self.enter_node
        queue = [(euclidean_dist(query_vec, self.nodes[start]), start)]
        visited.add(start)

        while queue:
            d, node = heapq.heappop(queue)
            candidates.append((d, node))
            for n in self.layers[0][node]:
                if n not in visited:
                    visited.add(n)
                    nd = euclidean_dist(query_vec, self.nodes[n])
                    heapq.heappush(queue, (nd, n))
            if len(candidates) >= self.ef_search:
                break

        candidates.sort(key=lambda x: x[0])
        return [{"id": nid, "distance": round(d, 4)} for d, nid in candidates[:top_k]]
