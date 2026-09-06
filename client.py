"""
Hierarchical Navigable Small World (HNSW) Vector Graph Index.
Zero external dependencies, standard library only.
"""

import math
import random
from typing import Dict, List, Any, Optional, Tuple

class HNSWVectorIndexClient:
    """
    Implements Hierarchical Navigable Small World (HNSW) graph indexing:
    - Multi-layer skip list graph representation
    - Cosine similarity distance metric
    - Greedy beam search across hierarchical layers
    """

    def __init__(self, m: int = 4, ef_construction: int = 16, ml: float = 0.62):
        self.m = m
        self.ef_construction = ef_construction
        self.ml = ml
        self.vectors = {} # id -> list of float
        self.layers = []  # layer_idx -> dict of node_id -> list of neighbor_ids
        self.entry_point = None

    def _cosine_similarity(self, v1: List[float], v2: List[float]) -> float:
        dot = sum(a * b for a, b in zip(v1, v2))
        norm1 = math.sqrt(sum(a * a for a in v1))
        norm2 = math.sqrt(sum(b * b for b in v2))
        if norm1 == 0.0 or norm2 == 0.0:
            return 0.0
        return dot / (norm1 * norm2)

    def _assign_random_level(self) -> int:
        r = random.random()
        return int(-math.log(r) * self.ml) if r > 0 else 0

    def add_vector(self, node_id: str, vector: List[float]):
        """Inserts vector into multi-layer HNSW graph."""
        self.vectors[node_id] = vector
        node_level = self._assign_random_level()
        old_max_level = len(self.layers) - 1

        while len(self.layers) <= node_level:
            self.layers.append({})

        if self.entry_point is None:
            self.entry_point = node_id
            for l in range(node_level + 1):
                self.layers[l][node_id] = []
            return

        curr_ep = self.entry_point
        # Search down from top layer to node_level + 1
        for l in range(old_max_level, node_level, -1):
            curr_ep = self._search_layer_greedy(curr_ep, vector, l)

        # Connect at layers <= min(node_level, old_max_level)
        for l in range(min(node_level, old_max_level), -1, -1):
            self.layers[l][node_id] = []
            neighbors = self._search_layer_knn(curr_ep, vector, self.ef_construction, l)
            selected = neighbors[:self.m]
            for n_id, _ in selected:
                if n_id != node_id:
                    self.layers[l][node_id].append(n_id)
                    if n_id in self.layers[l]:
                        self.layers[l][n_id].append(node_id)
            curr_ep = selected[0][0] if selected else curr_ep

        for l in range(old_max_level + 1, node_level + 1):
            self.layers[l][node_id] = []

        if node_level > old_max_level:
            self.entry_point = node_id

    def _search_layer_greedy(self, ep: str, query: List[float], layer_idx: int) -> str:
        if ep not in self.layers[layer_idx]:
            if self.layers[layer_idx]:
                ep = next(iter(self.layers[layer_idx]))
            else:
                return ep
        best_node = ep
        best_sim = self._cosine_similarity(query, self.vectors[ep])
        while True:
            changed = False
            for neighbor in self.layers[layer_idx].get(best_node, []):
                sim = self._cosine_similarity(query, self.vectors[neighbor])
                if sim > best_sim:
                    best_sim = sim
                    best_node = neighbor
                    changed = True
            if not changed:
                break
        return best_node

    def _search_layer_knn(self, ep: str, query: List[float], ef: int, layer_idx: int) -> List[Tuple[str, float]]:
        if ep not in self.layers[layer_idx]:
            if self.layers[layer_idx]:
                ep = next(iter(self.layers[layer_idx]))
            else:
                return []
        visited = {ep}
        candidates = [(ep, self._cosine_similarity(query, self.vectors[ep]))]
        w = list(candidates)

        while candidates:
            candidates.sort(key=lambda x: x[1], reverse=True)
            c_node, c_sim = candidates.pop(0)

            w.sort(key=lambda x: x[1], reverse=True)
            if c_sim < w[-1][1] and len(w) >= ef:
                break

            for neighbor in self.layers[layer_idx].get(c_node, []):
                if neighbor not in visited:
                    visited.add(neighbor)
                    n_sim = self._cosine_similarity(query, self.vectors[neighbor])
                    w.sort(key=lambda x: x[1], reverse=True)
                    if n_sim > w[-1][1] or len(w) < ef:
                        candidates.append((neighbor, n_sim))
                        w.append((neighbor, n_sim))
                        if len(w) > ef:
                            w.sort(key=lambda x: x[1], reverse=True)
                            w.pop()

        w.sort(key=lambda x: x[1], reverse=True)
        return w

    def search_knn(self, query: List[float], top_k: int = 3) -> List[Dict[str, Any]]:
        """Queries HNSW index for top_k closest vectors."""
        if not self.entry_point:
            return []

        curr_ep = self.entry_point
        for l in range(len(self.layers) - 1, 0, -1):
            curr_ep = self._search_layer_greedy(curr_ep, query, l)

        candidates = self._search_layer_knn(curr_ep, query, max(top_k * 2, self.ef_construction), 0)
        return [{"id": nid, "similarity": round(sim, 4)} for nid, sim in candidates[:top_k]]
