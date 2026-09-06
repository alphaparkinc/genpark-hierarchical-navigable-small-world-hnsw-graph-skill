"""
MCP Server for genpark-hierarchical-navigable-small-world-hnsw-graph-skill
Standard JSON-RPC 2.0 protocol over stdio.
"""

import sys
import json
from client import HNSWVectorIndexClient

index = HNSWVectorIndexClient()

def handle_request(req):
    req_id = req.get("id")
    method = req.get("method")
    params = req.get("params", {})

    if method == "tools/list":
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "result": {
                "tools": [
                    {
                        "name": "search_knn",
                        "description": "Perform Approximate Nearest Neighbor vector search via HNSW.",
                        "inputSchema": {
                            "type": "object",
                            "properties": {
                                "query": {"type": "array"},
                                "top_k": {"type": "integer"}
                            },
                            "required": ["query"]
                        }
                    }
                ]
            }
        }
    elif method == "tools/call":
        tool_name = params.get("name")
        args = params.get("arguments", {})
        if tool_name == "search_knn":
            res = index.search_knn(args.get("query", []), args.get("top_k", 3))
            return {"jsonrpc": "2.0", "id": req_id, "result": {"content": [{"type": "text", "text": json.dumps(res, indent=2)}]}}
    return {"jsonrpc": "2.0", "id": req_id, "error": {"code": -32601, "message": "Method not found"}}

def main():
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
            resp = handle_request(req)
            sys.stdout.write(json.dumps(resp) + "\n")
            sys.stdout.flush()
        except Exception as e:
            err = {"jsonrpc": "2.0", "id": None, "error": {"code": -32700, "message": str(e)}}
            sys.stdout.write(json.dumps(err) + "\n")
            sys.stdout.flush()

if __name__ == "__main__":
    main()
