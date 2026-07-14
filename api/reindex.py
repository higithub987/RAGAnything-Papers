"""
One-time migration: reads all existing entities, relationships, and text chunks
from rag_storage and upserts them into Milvus. Run this once to populate Milvus
with documents that were originally processed with NanoVectorDB.

Usage:
    wsl.exe bash -c "cd '...' && source .venv/bin/activate && python -m api.reindex"
"""

import asyncio
import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from lightrag.utils import compute_mdhash_id

from .config import settings
from .rag_manager import ensure_rag_ready, get_rag, initialize_rag

NS = "http://graphml.graphdrawing.org/xmlns"
BATCH_SIZE = 50


def _parse_element_data(element, key_map: dict) -> dict:
    return {
        key_map[d.get("key")]: d.text
        for d in element.iter(f"{{{NS}}}data")
        if d.get("key") in key_map
    }


def parse_graphml() -> tuple[dict, dict]:
    path = Path(settings.working_dir) / "graph_chunk_entity_relation.graphml"
    root = ET.parse(path).getroot()
    key_map = {k.get("id"): k.get("attr.name") for k in root.iter(f"{{{NS}}}key")}

    entities: dict = {}
    for node in root.iter(f"{{{NS}}}node"):
        data = _parse_element_data(node, key_map)
        name = node.get("id")
        if not data.get("description"):
            continue
        vdb_id = compute_mdhash_id(name, prefix="ent-")
        entities[vdb_id] = {
            "content": f"{name}\n{data['description']}",
            "entity_name": name,
            "entity_type": data.get("entity_type", ""),
            "description": data.get("description", ""),
            "source_id": data.get("source_id", ""),
            "file_path": data.get("file_path", ""),
        }

    relationships: dict = {}
    for edge in root.iter(f"{{{NS}}}edge"):
        src, tgt = edge.get("source"), edge.get("target")
        data = _parse_element_data(edge, key_map)
        if not data.get("description"):
            continue
        keywords = data.get("keywords", "")
        vdb_id = compute_mdhash_id(src + tgt, prefix="rel-")
        relationships[vdb_id] = {
            "content": f"{keywords}\t{src}\n{tgt}\n{data['description']}",
            "src_id": src,
            "tgt_id": tgt,
            "keywords": keywords,
            "description": data.get("description", ""),
            "weight": float(data.get("weight") or 1.0),
            "source_id": data.get("source_id", ""),
            "file_path": data.get("file_path", ""),
        }

    return entities, relationships


async def upsert_batched(vdb, data: dict, label: str) -> None:
    items = list(data.items())
    for i in range(0, len(items), BATCH_SIZE):
        batch = dict(items[i : i + BATCH_SIZE])
        await vdb.upsert(batch)
        done = min(i + BATCH_SIZE, len(items))
        print(f"  {label}: {done}/{len(items)}")


async def main() -> None:
    print("Initializing RAGAnything + LightRAG...")
    initialize_rag()
    await ensure_rag_ready()
    lg = get_rag().lightrag

    print("\n[1/3] Indexing text chunks...")
    chunks_path = Path(settings.working_dir) / "kv_store_text_chunks.json"
    chunks = json.loads(chunks_path.read_text(encoding="utf-8"))
    await upsert_batched(lg.chunks_vdb, chunks, "chunks")
    print(f"  Done — {len(chunks)} chunks indexed")

    print("\n[2/3] Parsing knowledge graph...")
    entities, relationships = parse_graphml()

    print(
        f"\n[3/3] Indexing {len(entities)} entities and {len(relationships)} relationships..."
    )
    await asyncio.gather(
        upsert_batched(lg.entities_vdb, entities, "entities"),
        upsert_batched(lg.relationships_vdb, relationships, "relationships"),
    )

    print("\nReindex complete.")
    await get_rag().finalize_storages()


if __name__ == "__main__":
    asyncio.run(main())
