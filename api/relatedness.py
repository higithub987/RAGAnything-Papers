from typing import Optional

import numpy as np

from .models import DocumentRelatedness, DocumentTopics, RelationDetail, TaskStatus, TopicDetail
from .rag_manager import get_rag
from .task_store import list_tasks


async def compute_relatedness() -> list[DocumentRelatedness]:
    """Pairwise cosine similarity between documents' mean chunk embeddings.

    Reuses the chunk embeddings LightRAG already computed during ingestion
    (via Milvus) instead of making new embedding calls -- a document's
    vector is just the average of its chunks' vectors.
    """
    tasks = [
        t for t in list_tasks() if t.status == TaskStatus.COMPLETED and t.doc_id
    ]
    if len(tasks) < 2:
        return []

    lightrag = get_rag().lightrag
    doc_statuses = await lightrag.aget_docs_by_ids([t.doc_id for t in tasks])

    all_chunk_ids = sorted(
        {
            chunk_id
            for status in doc_statuses.values()
            for chunk_id in (status.get("chunks_list") or [])
        }
    )
    vectors_by_chunk = await lightrag.chunks_vdb.get_vectors_by_ids(all_chunk_ids)

    doc_vectors: dict[str, np.ndarray] = {}
    for task in tasks:
        status = doc_statuses.get(task.doc_id)
        if status is None:
            continue
        chunk_vectors = [
            vectors_by_chunk[chunk_id]
            for chunk_id in (status.get("chunks_list") or [])
            if chunk_id in vectors_by_chunk
        ]
        if chunk_vectors:
            doc_vectors[task.doc_id] = np.mean(np.array(chunk_vectors), axis=0)

    doc_ids = [t.doc_id for t in tasks if t.doc_id in doc_vectors]
    file_names = {t.doc_id: t.file_name for t in tasks}

    entity_results = await lightrag.full_entities.get_by_ids(doc_ids)
    entities_by_doc = {
        doc_id: set((entry or {}).get("entity_names") or [])
        for doc_id, entry in zip(doc_ids, entity_results)
    }

    relation_results = await lightrag.full_relations.get_by_ids(doc_ids)
    relations_by_doc = {
        doc_id: [tuple(pair) for pair in (entry or {}).get("relation_pairs") or []]
        for doc_id, entry in zip(doc_ids, relation_results)
    }

    # First pass: scores + which shared topics/relations to show per pair, without
    # touching the graph yet -- collected into global sets so the (potentially many,
    # since pairs grow quadratically with doc count) per-pair lookups become two
    # bulk graph calls total instead of one pair of calls per pair.
    pair_data = []
    all_shared_entities: set[str] = set()
    all_relation_candidates: set[tuple[str, str]] = set()

    for i, doc_a in enumerate(doc_ids):
        vec_a = doc_vectors[doc_a]
        norm_a = np.linalg.norm(vec_a)
        for doc_b in doc_ids[i + 1 :]:
            vec_b = doc_vectors[doc_b]
            norm_b = np.linalg.norm(vec_b)
            if norm_a == 0 or norm_b == 0:
                continue
            score = float(np.dot(vec_a, vec_b) / (norm_a * norm_b))

            shared_entities = entities_by_doc.get(doc_a, set()) & entities_by_doc.get(
                doc_b, set()
            )
            top_shared = sorted(shared_entities)[:5]

            candidate_pairs = relations_by_doc.get(doc_a, []) + relations_by_doc.get(
                doc_b, []
            )
            relevant_relations = {
                (src, tgt) if src <= tgt else (tgt, src)
                for src, tgt in candidate_pairs
                if src in shared_entities and tgt in shared_entities
            }
            top_relations = sorted(relevant_relations)[:3]

            all_shared_entities.update(top_shared)
            all_relation_candidates.update(top_relations)

            pair_data.append(
                {
                    "doc_a": doc_a,
                    "doc_b": doc_b,
                    "score": score,
                    "top_shared": top_shared,
                    "top_relations": top_relations,
                }
            )

    graph = lightrag.chunk_entity_relation_graph
    node_data = (
        await graph.get_nodes_batch(sorted(all_shared_entities))
        if all_shared_entities
        else {}
    )
    edge_data = (
        await graph.get_edges_batch(
            [{"src": src, "tgt": tgt} for src, tgt in all_relation_candidates]
        )
        if all_relation_candidates
        else {}
    )

    def topic_detail(name: str) -> TopicDetail:
        info = node_data.get(name) or {}
        return TopicDetail(
            name=name,
            entity_type=info.get("entity_type", ""),
            description=info.get("description", ""),
        )

    def relation_detail(src: str, tgt: str) -> RelationDetail:
        info = edge_data.get((src, tgt)) or edge_data.get((tgt, src)) or {}
        return RelationDetail(
            source=src, target=tgt, description=info.get("description", "")
        )

    results = [
        DocumentRelatedness(
            doc_id=pd["doc_a"],
            file_name=file_names[pd["doc_a"]],
            related_doc_id=pd["doc_b"],
            related_file_name=file_names[pd["doc_b"]],
            score=pd["score"],
            shared_topics=[topic_detail(name) for name in pd["top_shared"]],
            shared_relations=[
                relation_detail(src, tgt) for src, tgt in pd["top_relations"]
            ],
        )
        for pd in pair_data
    ]

    results.sort(key=lambda r: r.score, reverse=True)
    return results


async def get_document_topics(doc_id: str) -> Optional[DocumentTopics]:
    """A single document's own extracted topics, for the node-click detail popover."""
    tasks = [t for t in list_tasks() if t.doc_id == doc_id]
    if not tasks:
        return None
    file_name = tasks[0].file_name

    lightrag = get_rag().lightrag
    entity_result = await lightrag.full_entities.get_by_id(doc_id)
    entity_names = sorted((entity_result or {}).get("entity_names") or [])[:15]
    if not entity_names:
        return DocumentTopics(doc_id=doc_id, file_name=file_name, topics=[])

    node_data = await lightrag.chunk_entity_relation_graph.get_nodes_batch(entity_names)
    topics = [
        TopicDetail(
            name=name,
            entity_type=(node_data.get(name) or {}).get("entity_type", ""),
            description=(node_data.get(name) or {}).get("description", ""),
        )
        for name in entity_names
    ]
    return DocumentTopics(doc_id=doc_id, file_name=file_name, topics=topics)
