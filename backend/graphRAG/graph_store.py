"""
Graph Store (Neo4j)
====================
Manages the Neo4j knowledge graph: storing triplets, querying subgraphs
for RAG retrieval, and exporting graph data for frontend visualisation.

Node schema:
    (:Entity {name: str, type: str, doc_id: str, ...props})

Relationship schema:
    [:RELATION {type: str, doc_id: str, source_page: int, source_chunk: int, confidence: float, ...props}]
"""

import logging
from typing import List, Optional, Dict, Any

from neo4j import GraphDatabase

from .config import NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD
from .triplet_extractor import Triplet

logger = logging.getLogger(__name__)

# ── Driver singleton ────────────────────────────────────
_driver = None


def _get_driver():
    global _driver
    if _driver is None:
        logger.info("Connecting to Neo4j at %s ...", NEO4J_URI)
        _driver = GraphDatabase.driver(
            NEO4J_URI,
            auth=(NEO4J_USERNAME, NEO4J_PASSWORD),
        )
        # Verify connectivity
        _driver.verify_connectivity()
        logger.info("Neo4j connection established.")
    return _driver


def close_driver():
    """Call on shutdown to cleanly close the driver."""
    global _driver
    if _driver is not None:
        _driver.close()
        _driver = None


# ── Index creation (run once) ───────────────────────────
def ensure_indexes():
    """Create indexes for efficient lookups."""
    driver = _get_driver()
    with driver.session() as session:
        # Index on Entity name + doc_id for fast lookups
        session.run(
            "CREATE INDEX entity_name_idx IF NOT EXISTS "
            "FOR (e:Entity) ON (e.name)"
        )
        session.run(
            "CREATE INDEX entity_doc_idx IF NOT EXISTS "
            "FOR (e:Entity) ON (e.doc_id)"
        )
        logger.info("Neo4j indexes ensured.")


# ── Store triplets ──────────────────────────────────────
def store_triplets(doc_id: str, triplets: List[Triplet]) -> int:
    """
    Store extracted triplets into Neo4j.
    Uses MERGE to avoid duplicates.  Returns number of relationships created.
    """
    if not triplets:
        return 0

    driver = _get_driver()
    ensure_indexes()

    count = 0
    with driver.session() as session:
        for t in triplets:
            try:
                subject_props = _sanitize_props(t.subject_props)
                object_props = _sanitize_props(t.object_props)
                relation_props = _sanitize_props(t.relation_props)
                session.run(
                    """
                    MERGE (s:Entity {name: $subject, doc_id: $doc_id})
                    ON CREATE SET s.type = $subject_type
                    SET s += $subject_props
                    MERGE (o:Entity {name: $object, doc_id: $doc_id})
                    ON CREATE SET o.type = $object_type
                    SET o += $object_props
                    MERGE (s)-[r:RELATION {type: $predicate, doc_id: $doc_id}]->(o)
                    ON CREATE SET r.source_page = $source_page,
                                  r.source_chunk = $source_chunk,
                                  r.confidence = $confidence
                    SET r += $relation_props
                    """,
                    subject=t.subject,
                    subject_type=t.subject_type,
                    subject_props=subject_props,
                    object=t.object,
                    object_type=t.object_type,
                    object_props=object_props,
                    predicate=t.predicate,
                    doc_id=doc_id,
                    source_page=t.source_page,
                    source_chunk=t.source_chunk,
                    confidence=t.confidence,
                    relation_props=relation_props,
                )
                count += 1
            except Exception as e:
                logger.warning("Failed to store triplet (%s)-[%s]->(%s): %s",
                               t.subject, t.predicate, t.object, e)

    logger.info("Stored %d triplets for doc_id=%s", count, doc_id)
    return count


# ── Query graph for RAG retrieval ───────────────────────
def _extract_query_tokens(entities: List[str]) -> List[str]:
    stop_words = {
        "the", "a", "an", "is", "are", "was", "were", "what", "how",
        "which", "who", "when", "where", "of", "in", "for", "to",
        "and", "or", "on", "at", "by", "with", "from", "about",
        "does", "do", "did", "has", "have", "had", "be", "been",
        "this", "that", "it", "its", "their", "they", "them",
    }

    tokens: List[str] = []
    for ent in entities:
        for part in (ent or "").replace("/", " ").split():
            word = "".join(ch for ch in part if ch.isalnum()).lower()
            if len(word) < 3 or word in stop_words:
                continue
            tokens.append(word)

    # De-duplicate while preserving order
    seen = set()
    deduped = []
    for t in tokens:
        if t not in seen:
            seen.add(t)
            deduped.append(t)
    return deduped


def query_graph(doc_id: str, entities: List[str], max_hops: int = 2) -> List[dict]:
    """
    Given a list of query entities, find them in the graph (fuzzy match)
    and traverse up to `max_hops` hops to gather context.

    Returns a list of triplet dicts:
        [{"subject": str, "predicate": str, "object": str,
          "subject_type": str, "object_type": str, "source_page": int}, ...]
    """
    if not entities:
        entities = []

    driver = _get_driver()

    # Build a regex pattern for fuzzy matching (case-insensitive CONTAINS)
    results = []
    seen = set()

    with driver.session() as session:
        for entity in entities:
            # Clean the entity name for Cypher
            safe_entity = entity.replace("'", "\\'").replace('"', '\\"')

            # Find nodes whose name contains the entity string (case-insensitive)
            # Then traverse up to max_hops
            query = """
            MATCH (e:Entity {doc_id: $doc_id})
            WHERE toLower(e.name) CONTAINS toLower($entity)
            CALL apoc.path.subgraphAll(e, {
                maxLevel: $max_hops,
                relationshipFilter: 'RELATION',
                labelFilter: '+Entity'
            })
            YIELD nodes, relationships
            UNWIND relationships AS r
            WITH startNode(r) AS s, r, endNode(r) AS o
            WHERE s.doc_id = $doc_id AND o.doc_id = $doc_id
              RETURN s.name AS subject, s.type AS subject_type,
                    properties(s) AS subject_props,
                    r.type AS predicate, r.source_page AS source_page,
                    properties(r) AS relation_props,
                    o.name AS object, o.type AS object_type,
                    properties(o) AS object_props
            LIMIT 50
            """

            # Simpler fallback query without APOC (more portable)
            fallback_query_1hop = """
            MATCH (e:Entity {doc_id: $doc_id})
            WHERE toLower(e.name) CONTAINS toLower($entity)
            MATCH (e)-[r:RELATION {doc_id: $doc_id}]-(neighbor:Entity {doc_id: $doc_id})
              RETURN e.name AS subject, e.type AS subject_type,
                    properties(e) AS subject_props,
                    r.type AS predicate, r.source_page AS source_page,
                    properties(r) AS relation_props,
                    neighbor.name AS object, neighbor.type AS object_type,
                    properties(neighbor) AS object_props
            LIMIT 50
            """

            fallback_query_2hop = """
            MATCH (e:Entity {doc_id: $doc_id})
            WHERE toLower(e.name) CONTAINS toLower($entity)
            MATCH (e)-[r1:RELATION {doc_id: $doc_id}]-(n1:Entity {doc_id: $doc_id})
            OPTIONAL MATCH (n1)-[r2:RELATION {doc_id: $doc_id}]-(n2:Entity {doc_id: $doc_id})
            WHERE n2 <> e
            WITH e, r1, n1, r2, n2
              RETURN e.name AS subject, e.type AS subject_type,
                    properties(e) AS subject_props,
                    r1.type AS predicate, r1.source_page AS source_page,
                    properties(r1) AS relation_props,
                    n1.name AS object, n1.type AS object_type,
                    properties(n1) AS object_props
            UNION
            MATCH (e:Entity {doc_id: $doc_id})
            WHERE toLower(e.name) CONTAINS toLower($entity)
            MATCH (e)-[r1:RELATION {doc_id: $doc_id}]-(n1:Entity {doc_id: $doc_id})
            MATCH (n1)-[r2:RELATION {doc_id: $doc_id}]-(n2:Entity {doc_id: $doc_id})
            WHERE n2 <> e
              RETURN n1.name AS subject, n1.type AS subject_type,
                    properties(n1) AS subject_props,
                    r2.type AS predicate, r2.source_page AS source_page,
                    properties(r2) AS relation_props,
                    n2.name AS object, n2.type AS object_type,
                    properties(n2) AS object_props
            LIMIT 50
            """

            try:
                # Try APOC first, fall back to simple Cypher
                try:
                    records = session.run(
                        query,
                        doc_id=doc_id, entity=entity, max_hops=max_hops,
                    ).data()
                except Exception:
                    # APOC not installed — use fallback
                    if max_hops >= 2:
                        records = session.run(
                            fallback_query_2hop,
                            doc_id=doc_id, entity=entity,
                        ).data()
                    else:
                        records = session.run(
                            fallback_query_1hop,
                            doc_id=doc_id, entity=entity,
                        ).data()

                for rec in records:
                    key = (rec.get("subject", ""), rec.get("predicate", ""), rec.get("object", ""))
                    if key not in seen:
                        seen.add(key)
                        results.append({
                            "subject": rec.get("subject", ""),
                            "subject_type": rec.get("subject_type", ""),
                            "subject_props": _sanitize_props(rec.get("subject_props", {})),
                            "predicate": rec.get("predicate", ""),
                            "relation_props": _sanitize_props(rec.get("relation_props", {})),
                            "object": rec.get("object", ""),
                            "object_type": rec.get("object_type", ""),
                            "object_props": _sanitize_props(rec.get("object_props", {})),
                            "source_page": rec.get("source_page", 0),
                        })
            except Exception as e:
                logger.warning("Graph query failed for entity '%s': %s", entity, e)

        if results:
            logger.info("Graph query for entities %s returned %d triplets", entities, len(results))
            return results

        # Fallback: keyword scan across subject/object/predicate for this doc_id
        tokens = _extract_query_tokens(entities)
        if tokens:
            fallback_keyword_query = """
            MATCH (s:Entity {doc_id: $doc_id})-[r:RELATION {doc_id: $doc_id}]-(o:Entity {doc_id: $doc_id})
            WHERE any(t IN $tokens WHERE
                toLower(s.name) CONTAINS t OR
                toLower(o.name) CONTAINS t OR
                toLower(r.type) CONTAINS t
            )
            RETURN s.name AS subject, s.type AS subject_type,
                   properties(s) AS subject_props,
                   r.type AS predicate, r.source_page AS source_page,
                   properties(r) AS relation_props,
                   o.name AS object, o.type AS object_type,
                   properties(o) AS object_props
            LIMIT 50
            """

            records = session.run(
                fallback_keyword_query,
                doc_id=doc_id,
                tokens=tokens,
            ).data()

            for rec in records:
                key = (rec.get("subject", ""), rec.get("predicate", ""), rec.get("object", ""))
                if key not in seen:
                    seen.add(key)
                    results.append({
                        "subject": rec.get("subject", ""),
                        "subject_type": rec.get("subject_type", ""),
                        "subject_props": _sanitize_props(rec.get("subject_props", {})),
                        "predicate": rec.get("predicate", ""),
                        "relation_props": _sanitize_props(rec.get("relation_props", {})),
                        "object": rec.get("object", ""),
                        "object_type": rec.get("object_type", ""),
                        "object_props": _sanitize_props(rec.get("object_props", {})),
                        "source_page": rec.get("source_page", 0),
                    })

            if results:
                logger.info(
                    "Graph keyword fallback for entities %s returned %d triplets",
                    entities,
                    len(results),
                )
                return results

        # Last-resort fallback: return top relations for this doc_id
        fallback_top_query = """
        MATCH (s:Entity {doc_id: $doc_id})-[r:RELATION {doc_id: $doc_id}]->(o:Entity {doc_id: $doc_id})
        RETURN s.name AS subject, s.type AS subject_type,
               properties(s) AS subject_props,
               r.type AS predicate, r.source_page AS source_page,
               properties(r) AS relation_props,
               o.name AS object, o.type AS object_type,
               properties(o) AS object_props
        ORDER BY r.confidence DESC
        LIMIT 50
        """
        records = session.run(fallback_top_query, doc_id=doc_id).data()
        for rec in records:
            key = (rec.get("subject", ""), rec.get("predicate", ""), rec.get("object", ""))
            if key not in seen:
                seen.add(key)
                results.append({
                    "subject": rec.get("subject", ""),
                    "subject_type": rec.get("subject_type", ""),
                    "subject_props": _sanitize_props(rec.get("subject_props", {})),
                    "predicate": rec.get("predicate", ""),
                    "relation_props": _sanitize_props(rec.get("relation_props", {})),
                    "object": rec.get("object", ""),
                    "object_type": rec.get("object_type", ""),
                    "object_props": _sanitize_props(rec.get("object_props", {})),
                    "source_page": rec.get("source_page", 0),
                })

        logger.info(
            "Graph fallback (top relations) for entities %s returned %d triplets",
            entities,
            len(results),
        )
        return results


# ── Get full document graph (for visualization) ────────
def get_document_graph(doc_id: str) -> Dict[str, Any]:
    """
    Return the complete graph for a document, formatted for vis.js.

    Returns:
        {
            "nodes": [{"id": int, "label": str, "type": str, "group": str}, ...],
            "edges": [{"from": int, "to": int, "label": str, "source_page": int}, ...],
            "stats": {"node_count": int, "edge_count": int}
        }
    """
    driver = _get_driver()

    with driver.session() as session:
        # Get all nodes for this document
        node_records = session.run(
            """
            MATCH (e:Entity {doc_id: $doc_id})
            RETURN id(e) AS neo_id, e.name AS name, e.type AS type
            ORDER BY e.name
            """,
            doc_id=doc_id,
        ).data()

        # Get all relationships for this document
        edge_records = session.run(
            """
            MATCH (s:Entity {doc_id: $doc_id})-[r:RELATION {doc_id: $doc_id}]->(o:Entity {doc_id: $doc_id})
            RETURN id(s) AS source_id, id(o) AS target_id,
                   r.type AS predicate, r.source_page AS source_page,
                   r.confidence AS confidence
            """,
            doc_id=doc_id,
        ).data()

    # Map Neo4j internal IDs to sequential vis.js IDs
    neo_to_vis = {}
    nodes = []
    for i, rec in enumerate(node_records):
        vis_id = i + 1
        neo_to_vis[rec["neo_id"]] = vis_id
        nodes.append({
            "id": vis_id,
            "label": rec["name"],
            "type": rec.get("type", "ENTITY"),
            "group": rec.get("type", "ENTITY"),
        })

    edges = []
    for rec in edge_records:
        source = neo_to_vis.get(rec["source_id"])
        target = neo_to_vis.get(rec["target_id"])
        if source and target:
            edges.append({
                "from": source,
                "to": target,
                "label": (rec.get("predicate") or "").replace("_", " ").title(),
                "source_page": rec.get("source_page", 0),
                "confidence": rec.get("confidence", 1.0),
            })

    return {
        "nodes": nodes,
        "edges": edges,
        "stats": {
            "node_count": len(nodes),
            "edge_count": len(edges),
        },
    }


def _sanitize_props(props: Dict[str, Any]) -> Dict[str, Any]:
    """Remove reserved keys and non-serializable values."""
    if not isinstance(props, dict):
        return {}
    reserved = {"name", "doc_id", "type"}
    cleaned = {}
    for k, v in props.items():
        if k in reserved:
            continue
        if isinstance(v, (str, int, float, bool)) or v is None:
            cleaned[k] = v
    return cleaned


# ── Delete document graph ──────────────────────────────
def delete_document_graph(doc_id: str) -> int:
    """Remove all nodes and relationships for a document. Returns count deleted."""
    driver = _get_driver()

    with driver.session() as session:
        result = session.run(
            """
            MATCH (e:Entity {doc_id: $doc_id})
            DETACH DELETE e
            RETURN count(e) AS deleted
            """,
            doc_id=doc_id,
        ).single()

    count = result["deleted"] if result else 0
    logger.info("Deleted %d nodes for doc_id=%s", count, doc_id)
    return count


# ── List documents with graphs ─────────────────────────
def list_graph_documents() -> List[dict]:
    """Return a list of doc_ids that have graph data, with node/edge counts."""
    driver = _get_driver()

    with driver.session() as session:
        records = session.run(
            """
            MATCH (e:Entity)
            WITH e.doc_id AS doc_id, count(e) AS node_count
            OPTIONAL MATCH (s:Entity {doc_id: doc_id})-[r:RELATION {doc_id: doc_id}]->(o:Entity)
            WITH doc_id, node_count, count(r) AS edge_count
            RETURN doc_id, node_count, edge_count
            ORDER BY doc_id
            """
        ).data()

    return [
        {
            "doc_id": r["doc_id"],
            "node_count": r["node_count"],
            "edge_count": r["edge_count"],
        }
        for r in records
    ]
