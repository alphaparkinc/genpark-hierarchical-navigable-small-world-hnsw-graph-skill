from client import HNSWGraph

hnsw = HNSWGraph()
hnsw.insert("doc1", [1.0, 0.0, 0.0])
hnsw.insert("doc2", [0.9, 0.1, 0.0])
hnsw.insert("doc3", [0.0, 1.0, 0.0])
results = hnsw.search([1.0, 0.0, 0.0], top_k=2)
print("ANN Results:", results)
