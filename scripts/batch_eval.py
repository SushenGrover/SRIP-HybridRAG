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


def load_questions(path):
    questions = []
    with open(path, "r", encoding="utf-8") as handle:
        for raw in handle:
            line = raw.strip()
            if not line:
                continue
            if line.startswith("*") and line.endswith("*"):
                continue
            questions.append(line)
    return questions


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


def main():
    parser = argparse.ArgumentParser(description="Batch query VectorRAG/GraphRAG/HybridRAG.")
    parser.add_argument("--questions", default="ques.txt", help="Path to questions text file")
    parser.add_argument("--filename", required=True, help="Document filename to match")
    parser.add_argument("--output", default="answers.csv", help="CSV output path")
    parser.add_argument("--vector-base", default="http://localhost:8000")
    parser.add_argument("--graph-base", default="http://localhost:8001")
    parser.add_argument("--hybrid-base", default="http://localhost:8002")
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    questions = load_questions(args.questions)
    if not questions:
        raise RuntimeError("No questions found in the input file")

    doc_id = resolve_doc_id(args.vector_base, args.filename)

    rows = []
    for q in questions:
        vector_answer = ""
        graph_answer = ""
        hybrid_answer = ""

        try:
            vector_resp = http_json(
                f"{args.vector_base}/api/query",
                method="POST",
                payload={
                    "document_id": doc_id,
                    "query": q,
                    "top_k": args.top_k,
                },
            )
            vector_answer = (vector_resp or {}).get("answer", "")
        except Exception as exc:
            print(f"VectorRAG failed for: {q} -> {exc}", file=sys.stderr)

        try:
            graph_resp = http_json(
                f"{args.graph_base}/api/graph/query",
                method="POST",
                payload={
                    "document_id": doc_id,
                    "query": q,
                },
            )
            graph_answer = (graph_resp or {}).get("answer", "")
        except Exception as exc:
            print(f"GraphRAG failed for: {q} -> {exc}", file=sys.stderr)

        try:
            hybrid_resp = http_json(
                f"{args.hybrid_base}/api/hybrid/compose",
                method="POST",
                payload={
                    "query": q,
                    "vector_answer": vector_answer,
                    "graph_answer": graph_answer,
                },
            )
            hybrid_answer = (hybrid_resp or {}).get("answer", "")
        except Exception as exc:
            print(f"HybridRAG failed for: {q} -> {exc}", file=sys.stderr)

        rows.append({
            "question": q,
            "vectorRAG_answer": vector_answer,
            "graphRAG_answer": graph_answer,
            "hybridRAG_answer": hybrid_answer,
        })

    with open(args.output, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "question",
                "vectorRAG_answer",
                "graphRAG_answer",
                "hybridRAG_answer",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} rows to {args.output}")


if __name__ == "__main__":
    main()
