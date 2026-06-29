"""
Startup/admin reconciliation for orphaned document storage.

A document that crashed mid-processing (server killed, OOM, etc.) before any
of processor.py's exception handlers ran leaves its doc_status stuck at
HANDLING/PROCESSING/PENDING with partial chunks/entities/vectors/graph nodes
already written. Unlike a clean exception path (handled by processor.py's
_rollback_document calls), nothing ever runs for a hard crash -- this module
scans for and cleans up exactly that class of leftover on the next clean
startup, or on demand via scripts/reconcile_storage.py.
"""

from __future__ import annotations

import logging

from raganything.base import DocStatus

logger = logging.getLogger(__name__)

# These statuses can only mean a previous process died mid-write: nothing
# legitimately survives a restart since pipeline state lives in memory.
_STALE_STATUSES = (
    DocStatus.HANDLING,
    DocStatus.PROCESSING,
    DocStatus.PENDING,
)


async def find_orphaned_doc_ids(lightrag) -> list[str]:
    """Return doc_ids stuck in a non-terminal status (HANDLING/PROCESSING/PENDING)."""
    stale_docs = await lightrag.doc_status.get_docs_by_statuses(list(_STALE_STATUSES))
    return list(stale_docs.keys())


async def reconcile_orphaned_documents(lightrag) -> dict[str, str]:
    """Roll back every orphaned doc_id found via adelete_by_doc_id.

    Returns a dict of {doc_id: outcome} where outcome is the DeletionResult
    status ("success"/"not_found"/etc.) or "error: <msg>" if the delete call
    itself raised. Best-effort -- one failed doc_id does not stop the sweep
    from attempting the rest.
    """
    orphans = await find_orphaned_doc_ids(lightrag)
    if not orphans:
        logger.info("Reconciliation sweep: no orphaned doc_ids found")
        return {}

    logger.warning(
        f"Reconciliation sweep: found {len(orphans)} orphaned doc_id(s) from a "
        f"prior crash/restart, rolling back: {orphans}"
    )
    outcomes: dict[str, str] = {}
    for doc_id in orphans:
        try:
            result = await lightrag.adelete_by_doc_id(doc_id)
            outcomes[doc_id] = result.status
            if result.status not in ("success", "not_found"):
                logger.error(
                    f"Reconciliation: doc_id={doc_id} rollback returned "
                    f"{result.status!r}: {result.message}"
                )
        except Exception as exc:
            logger.error(f"Reconciliation: doc_id={doc_id} rollback raised: {exc}")
            outcomes[doc_id] = f"error: {exc}"
    return outcomes
