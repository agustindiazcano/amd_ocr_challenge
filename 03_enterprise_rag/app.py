import argparse
import requests
import json
import os
import sys

DAEMON_URL = "http://127.0.0.1:8000"

def index_corpus(corpus_path: str):
    print(f"Sending index request for corpus: {corpus_path}")
    try:
        # Increase timeout heavily for index pass (up to 10 mins allowed by eval)
        response = requests.post(f"{DAEMON_URL}/index", json={"corpus_path": corpus_path}, timeout=600)
        response.raise_for_status()
        print("Indexing completed:", response.json())
    except Exception as e:
        print(f"Failed to index corpus: {e}", file=sys.stderr)

def run_query(corpus_path: str, query_id: str, query_text: str):
    # Evaluator expected output directory
    output_dir = "/app/output"
    output_file = os.path.join(output_dir, f"{query_id}_output.json")
    
    # 1. Base Fallback Structure (Strict Rule)
    result = {"answer": "", "citations": []}
    
    try:
        os.makedirs(output_dir, exist_ok=True)
        payload = {
            "query_id": query_id,
            "query": query_text,
            "corpus_path": corpus_path
        }
        
        # 30-sec limit per query by the evaluator. We use 29s to allow safe writing time.
        response = requests.post(f"{DAEMON_URL}/query", json=payload, timeout=29)
        response.raise_for_status() # Catches 500s or 400s
        
        data = response.json()
        result = {
            "answer": data.get("answer", ""),
            "citations": data.get("citations", []),
        }
            
    except Exception as e:
        # Global Catch-All: Timeout, Connection Refused, Daemon Crash
        print(f"Critical failure querying daemon (returning fallback empty result): {type(e).__name__} - {e}", file=sys.stderr)
    
    finally:
        # 2. Guarantee Writing
        # The finally block ensures that even if a fatal exception occurs above, 
        # the fallback {"answer": "", "citations": []} is written.
        try:
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
            print(f"Results safely written to {output_file}")
        except Exception as write_error:
            # Extreme edge case (e.g., completely out of disk space)
            print(f"FATAL: Could not write output file: {write_error}", file=sys.stderr)

def main():
    parser = argparse.ArgumentParser(description="Thin client for RAG evaluation")
    parser.add_argument("--index", type=str, help="Path to corpus to index")
    parser.add_argument("--corpus", type=str, help="Path to corpus (for querying)")
    parser.add_argument("--query-id", type=str, help="Unique ID for the query")
    parser.add_argument("--query", type=str, help="Text of the question")
    
    args = parser.parse_args()

    if args.index:
        index_corpus(args.index)
    elif args.query_id and args.query and args.corpus:
        run_query(args.corpus, args.query_id, args.query)
    else:
        parser.print_help()
        sys.exit(1)

if __name__ == "__main__":
    main()
