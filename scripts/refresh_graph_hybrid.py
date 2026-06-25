import argparse
import csv
import json
import sys
import urllib.error
import urllib.request


def http_json(url, method="GET", payload=None, timeout=120):
    data = None
    headers = {"Accept": "application/json"}
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"

    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = resp.read().decode("utf-8")
            if not body:
                return None
            return json.loads(body)
    except urllib.error.HTTPError as exc:
        try:
            err_body = exc.read().decode("utf-8")
        except Exception:
            err_body = ""
        raise RuntimeError(f"HTTP {exc.code} for {url}: {err_body}") from exc


def resolve_doc_id(vector_base, filename):
    data = http_json(f"{vector_base}/api/documents")
    docs = data.get("documents", []) if isinstance(data, dict) else []
    matches = [d for d in docs if d.get("filename", "").lower() == filename.lower()]
    if not matches:
        available = ", ".join([d.get("filename", "?") for d in docs])
        raise RuntimeError(
            f"No document found for filename '{filename}'. Available: {available}"
        )
    if len(matches) > 1:
        ids = ", ".join([d.get("document_id", "?") for d in matches])
        raise RuntimeError(
            f"Multiple documents matched '{filename}'. Document IDs: {ids}"
        )
    return matches[0].get("document_id")


WARNING_TEXT = (
    "⚠️ The requested information was not found in the knowledge graph. "
    "The graph may not contain this specific relationship."
)


def _needs_graph_refresh(graph_answer: str) -> bool:
    if not graph_answer:
        return True
    return graph_answer.strip() == WARNING_TEXT


def main():
    parser = argparse.ArgumentParser(
        description="Refresh GraphRAG + HybridRAG answers in an existing CSV.")
    parser.add_argument("--csv", default="answers.csv", help="Input CSV path")
    parser.add_argument("--output", help="Output CSV path (defaults to input)")
    parser.add_argument("--filename", required=True, help="Document filename to match")
    parser.add_argument("--vector-base", default="http://localhost:8000")
    parser.add_argument("--graph-base", default="http://localhost:8001")
    parser.add_argument("--hybrid-base", default="http://localhost:8002")
    args = parser.parse_args()

    doc_id = resolve_doc_id(args.vector_base, args.filename)

    output_path = args.output or args.csv

    try:
        handle = open(args.csv, "r", encoding="utf-8-sig", newline="")
        reader = csv.DictReader(handle)
        rows = list(reader)
        fieldnames = reader.fieldnames or []
        handle.close()
    except UnicodeDecodeError:
        handle = open(args.csv, "r", encoding="cp1252", newline="")
        reader = csv.DictReader(handle)
        rows = list(reader)
        fieldnames = reader.fieldnames or []
        handle.close()

    required = {"question", "vectorRAG_answer", "graphRAG_answer", "hybridRAG_answer"}
    missing = required.difference(fieldnames)
    if missing:
        raise RuntimeError(f"CSV missing required columns: {', '.join(sorted(missing))}")

    if "vectorRAG_sources" not in fieldnames:
        fieldnames.append("vectorRAG_sources")
    if "graphRAG_triplets" not in fieldnames:
        fieldnames.append("graphRAG_triplets")

    for row in rows:
        question = (row.get("question") or "").strip()
        if not question:
            continue

        vector_answer = row.get("vectorRAG_answer", "")
        graph_answer = row.get("graphRAG_answer", "")
        hybrid_answer = row.get("hybridRAG_answer", "")

        if _needs_graph_refresh(graph_answer):
            try:
                graph_resp = http_json(
                    f"{args.graph_base}/api/graph/query",
                    method="POST",
                    payload={
                        "document_id": doc_id,
                        "query": question,
                    },
                )
                graph_answer = (graph_resp or {}).get("answer", graph_answer)
                print(f"Updated GraphRAG answer for: {question}")
                print(f"  GraphRAG: {graph_answer}")
            except Exception as exc:
                print(f"GraphRAG failed for: {question} -> {exc}", file=sys.stderr)

        try:
            vector_retrieve = http_json(
                f"{args.vector_base}/api/retrieve",
                method="POST",
                payload={
                    "document_id": doc_id,
                    "query": question,
                    "top_k": 5,
                },
            )
            row["vectorRAG_sources"] = json.dumps(
                (vector_retrieve or {}).get("sources", []),
                ensure_ascii=False,
            )
        except Exception as exc:
            print(f"VectorRAG retrieve failed for: {question} -> {exc}", file=sys.stderr)

        try:
            graph_retrieve = http_json(
                f"{args.graph_base}/api/graph/retrieve",
                method="POST",
                payload={
                    "document_id": doc_id,
                    "query": question,
                },
            )
            row["graphRAG_triplets"] = json.dumps(
                (graph_retrieve or {}).get("triplets_used", []),
                ensure_ascii=False,
            )
        except Exception as exc:
            print(f"GraphRAG retrieve failed for: {question} -> {exc}", file=sys.stderr)

        try:
            hybrid_resp = http_json(
                f"{args.hybrid_base}/api/hybrid/compose",
                method="POST",
                payload={
                    "query": question,
                    "vector_answer": vector_answer,
                    "graph_answer": graph_answer,
                },
            )
            hybrid_answer = (hybrid_resp or {}).get("answer", hybrid_answer)
            print(f"Updated HybridRAG answer for: {question}")
            print(f"  HybridRAG: {hybrid_answer}")
        except Exception as exc:
            print(f"HybridRAG failed for: {question} -> {exc}", file=sys.stderr)

        row["graphRAG_answer"] = graph_answer
        row["hybridRAG_answer"] = hybrid_answer

    with open(output_path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Updated {len(rows)} rows in {output_path}")


if __name__ == "__main__":
    main()
