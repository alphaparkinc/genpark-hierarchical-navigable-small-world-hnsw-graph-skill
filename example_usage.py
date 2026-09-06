"""
Demonstration of genpark-hierarchical-navigable-small-world-hnsw-graph-skill
"""

from client import HNSWVectorIndexClient

def main():
    index = HNSWVectorIndexClient(m=3, ef_construction=8)

    # Insert 5 embedding vectors
    docs = {
        "doc_ai_agents": [0.95, 0.88, 0.12, 0.05],
        "doc_machine_learning": [0.90, 0.82, 0.15, 0.08],
        "doc_culinary_recipe": [0.05, 0.12, 0.94, 0.89],
        "doc_baking_bread": [0.08, 0.15, 0.91, 0.95],
        "doc_quantum_physics": [0.45, 0.40, 0.30, 0.25]
    }

    for doc_id, vec in docs.items():
        index.add_vector(doc_id, vec)

    # Query for AI agent vector
    query_vec = [0.92, 0.85, 0.10, 0.04]
    results = index.search_knn(query_vec, top_k=2)

    print("=== HNSW KNN VECTOR SEARCH RESULTS ===")
    for r in results:
        print(f"[{r['id']}] Cosine Similarity: {r['similarity']}")

if __name__ == "__main__":
    main()
