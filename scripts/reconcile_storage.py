"""
Manual/admin CLI to sweep orphaned doc_ids out of LightRAG storage.

Usage (run from the repo root, with the API's environment configured):
    python scripts/reconcile_storage.py [--dry-run]

Safe to run any time the server is not actively processing a document.
adelete_by_doc_id checks LightRAG's pipeline busy-flag and will refuse to
delete a doc_id while the pipeline is mid-job, so running this against a
live, idle server (e.g. via cron) cannot corrupt an in-flight insert.
"""

import argparse
import asyncio

from api.rag_manager import ensure_rag_ready, get_rag, initialize_rag
from raganything.reconciliation import (
    find_orphaned_doc_ids,
    reconcile_orphaned_documents,
)


async def main(dry_run: bool) -> None:
    initialize_rag()
    rag = get_rag()
    # Reconciliation only needs LightRAG's storages, never the document
    # parser -- skip that unrelated check so this still works on a host
    # where the parser binary isn't installed.
    rag._parser_installation_checked = True
    await ensure_rag_ready()
    lightrag = rag.lightrag

    if dry_run:
        orphans = await find_orphaned_doc_ids(lightrag)
        print(f"Would roll back {len(orphans)} orphaned doc_id(s): {orphans}")
        return

    outcomes = await reconcile_orphaned_documents(lightrag)
    for doc_id, outcome in outcomes.items():
        print(f"{doc_id}: {outcome}")
    print(f"Reconciled {len(outcomes)} doc_id(s)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    asyncio.run(main(args.dry_run))
