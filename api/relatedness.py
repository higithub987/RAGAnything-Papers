import numpy as np

from .models import DocumentRelatedness, TaskStatus
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
            for chunk_id in (status.chunks_list or [])
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
            for chunk_id in (status.chunks_list or [])
            if chunk_id in vectors_by_chunk
        ]
        if chunk_vectors:
            doc_vectors[task.doc_id] = np.mean(np.array(chunk_vectors), axis=0)

    results: list[DocumentRelatedness] = []
    doc_ids = [t.doc_id for t in tasks if t.doc_id in doc_vectors]
    file_names = {t.doc_id: t.file_name for t in tasks}
    for i, doc_a in enumerate(doc_ids):
        vec_a = doc_vectors[doc_a]
        norm_a = np.linalg.norm(vec_a)
        for doc_b in doc_ids[i + 1 :]:
            vec_b = doc_vectors[doc_b]
            norm_b = np.linalg.norm(vec_b)
            if norm_a == 0 or norm_b == 0:
                continue
            score = float(np.dot(vec_a, vec_b) / (norm_a * norm_b))
            results.append(
                DocumentRelatedness(
                    doc_id=doc_a,
                    file_name=file_names[doc_a],
                    related_doc_id=doc_b,
                    related_file_name=file_names[doc_b],
                    score=score,
                )
            )

    results.sort(key=lambda r: r.score, reverse=True)
    return results
