import sys
import json
from client import HNSWGraph

hnsw = HNSWGraph()

def handle_request(req):
    method = req.get("method")
    req_id = req.get("id")
    
    if method == "initialize":
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "result": {
                "protocolVersion": "2024-11-05",
                "capabilities": {"tools": {}},
                "serverInfo": {"name": "genpark-hierarchical-navigable-small-world-hnsw-graph-skill", "version": "1.0.0"}
            }
        }
    elif method == "tools/list":
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "result": {
                "tools": [
                    {
                        "name": "insert_vector",
                        "description": "Inserts vector into HNSW index",
                        "inputSchema": {
                            "type": "object",
                            "properties": {
                                "id": {"type": "string"},
                                "vector": {"type": "array", "items": {"type": "number"}}
                            },
                            "required": ["id", "vector"]
                        }
                    },
                    {
                        "name": "search_vector",
                        "description": "Performs approximate nearest neighbor search over HNSW graph",
                        "inputSchema": {
                            "type": "object",
                            "properties": {
                                "query_vector": {"type": "array", "items": {"type": "number"}},
                                "top_k": {"type": "integer", "default": 3}
                            },
                            "required": ["query_vector"]
                        }
                    }
                ]
            }
        }
    elif method == "tools/call":
        params = req.get("params", {})
        name = params.get("name")
        args = params.get("arguments", {})
        
        if name == "insert_vector":
            hnsw.insert(args.get("id", ""), args.get("vector", []))
            return {"jsonrpc": "2.0", "id": req_id, "result": {"content": [{"type": "text", "text": "Inserted"}]}}
        elif name == "search_vector":
            res = hnsw.search(args.get("query_vector", []), args.get("top_k", 3))
            return {"jsonrpc": "2.0", "id": req_id, "result": {"content": [{"type": "text", "text": json.dumps(res, indent=2)}]}}
            
    return {"jsonrpc": "2.0", "id": req_id, "error": {"code": -32601, "message": "Method not found"}}

def run():
    for line in sys.stdin:
        if not line.strip():
            continue
        try:
            req = json.loads(line)
            res = handle_request(req)
            sys.stdout.write(json.dumps(res) + "\n")
            sys.stdout.flush()
        except Exception as e:
            sys.stdout.write(json.dumps({"jsonrpc": "2.0", "error": {"code": -32000, "message": str(e)}}) + "\n")
            sys.stdout.flush()

if __name__ == "__main__":
    run()
