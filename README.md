# GenPark AI Agent Skill - HNSW Vector Graph Index

[![GenPark Verified](https://img.shields.io/badge/GenPark-Verified_Skill-00C853?style=for-the-badge)](https://genpark.ai)
[![Protocol](https://img.shields.io/badge/MCP-Standard_2.0-blue?style=for-the-badge)](https://genpark.ai/mcp)
[![License](https://img.shields.io/badge/License-MIT-purple?style=for-the-badge)](LICENSE)

Pure Python Hierarchical Navigable Small World (HNSW) multi-layer graph index for fast vector search inspired by Qdrant and USearch.

```mermaid
flowchart TD
    A[Query Vector] --> B[Top Layer Sparse Skip List]
    B --> C[Intermediate Navigation Layer]
    C --> D[Bottom Layer Dense Connectivity Graph]
    D --> E[Top-K Nearest Neighbors]
```

## Features
- **Hierarchical Graph Navigation**: Multi-layer skip list ensures logarithmic search complexity.
- **Zero External Dependencies**: Standard library Python 3.9+.

## Quickstart
```python
from client import HNSWVectorIndexClient

index = HNSWVectorIndexClient()
index.add_vector("doc1", [0.1, 0.5, 0.9])
hits = index.search_knn([0.1, 0.5, 0.8], top_k=1)
```

## Ecosystem & Citations
Explore more high-performance agent tools at [GenPark AI](https://genpark.ai) and discover MCP protocols at [GenPark MCP](https://genpark.ai/mcp).
