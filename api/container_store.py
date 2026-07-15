"""Durable registry of containers ("databases") and their document membership.

Part 1 of the container feature: the single source of truth for the
`container <-> document` mapping that every later layer (API routes, ingestion,
query scoping, the Visualize tab) reads from. This layer is self-contained --
no Milvus, no LightRAG, no running server -- so it can be unit-tested in
isolation.

Membership is **many-to-many**: a document (keyed by its opaque LightRAG doc_id)
may belong to several containers. The "All"/"query everything" view is
**virtual** -- it means "no container filter" and is deliberately NOT stored
here; the registry only holds real, user-created containers.

State is persisted to `<working_dir>/containers.json` because container
membership must survive a server restart (documents themselves persist via
LightRAG's on-disk doc_status, so their memberships must too). The forward map
(container -> docs) is the source of truth on disk; the reverse index
(doc -> containers) is derived in memory on load to avoid drift, mirroring how
`task_store` derives its path index.

Modeled on `relatedness_boost.py`: a lock-guarded class with an injectable
`working_dir` (for hermetic tests) exposed through a process-wide `get_registry()`
singleton.
"""

import json
import logging
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from .config import settings
from .models import Container

_log = logging.getLogger(__name__)

_FILENAME = "containers.json"


class ContainerRegistry:
    """Holds containers + their membership and persists them to JSON."""

    def __init__(self, working_dir: Optional[str] = None):
        base = Path(working_dir or settings.working_dir)
        self._path = base / _FILENAME
        self._lock = threading.Lock()
        # container_id -> Container (the on-disk source of truth)
        self._containers: dict[str, Container] = {}
        # doc_id -> {container_id, ...}; derived from _containers on load.
        self._doc_index: dict[str, set[str]] = {}
        self._load()

    # --- persistence -----------------------------------------------------
    def _load(self) -> None:
        """(Re)load state from disk. Tolerates a missing/corrupt file.

        Defense in depth (mirrors rag_manager.load_existing_documents): a
        bad/empty/partial read must never crash construction -- fall back to an
        empty registry so the process still starts.
        """
        if not self._path.exists():
            self._containers = {}
            self._doc_index = {}
            return
        try:
            text = self._path.read_text(encoding="utf-8")
            raw = json.loads(text) if text.strip() else {}
        except (OSError, json.JSONDecodeError):
            _log.warning("containers.json unreadable; starting with an empty registry")
            self._containers = {}
            self._doc_index = {}
            return

        containers: dict[str, Container] = {}
        for cid, entry in raw.items():
            try:
                container = Container(
                    id=entry["id"],
                    name=entry["name"],
                    created_at=datetime.fromisoformat(entry["created_at"]),
                    member_doc_ids=list(entry.get("member_doc_ids") or []),
                )
            except (KeyError, ValueError, TypeError):
                _log.warning("Skipping malformed container entry %r", cid)
                continue
            containers[container.id] = container

        self._containers = containers
        self._doc_index = self._build_doc_index(containers)

    @staticmethod
    def _build_doc_index(containers: dict[str, Container]) -> dict[str, set[str]]:
        index: dict[str, set[str]] = {}
        for container in containers.values():
            for doc_id in container.member_doc_ids:
                index.setdefault(doc_id, set()).add(container.id)
        return index

    def _save(self) -> None:
        """Persist the forward map. Caller must hold the lock."""
        data = {
            cid: {
                "id": c.id,
                "name": c.name,
                "created_at": c.created_at.isoformat(),
                "member_doc_ids": list(c.member_doc_ids),
            }
            for cid, c in self._containers.items()
        }
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.write_text(json.dumps(data), encoding="utf-8")

    # --- container CRUD --------------------------------------------------
    def create(self, name: str) -> Container:
        with self._lock:
            container = Container(
                id=str(uuid.uuid4()),
                name=name,
                created_at=datetime.now(timezone.utc),
                member_doc_ids=[],
            )
            self._containers[container.id] = container
            self._save()
            return container

    def get(self, container_id: str) -> Optional[Container]:
        with self._lock:
            return self._containers.get(container_id)

    def list_containers(self) -> list[Container]:
        """All containers, oldest-first (stable, so ties keep creation order)."""
        with self._lock:
            return sorted(self._containers.values(), key=lambda c: c.created_at)

    def rename(self, container_id: str, name: str) -> Optional[Container]:
        with self._lock:
            container = self._containers.get(container_id)
            if container is None:
                return None
            container.name = name
            self._save()
            return container

    def delete(self, container_id: str) -> bool:
        """Remove a container and drop it from every doc's membership."""
        with self._lock:
            container = self._containers.pop(container_id, None)
            if container is None:
                return False
            for doc_id in container.member_doc_ids:
                self._detach(doc_id, container_id)
            self._save()
            return True

    # --- membership (many-to-many) --------------------------------------
    def add_document(self, container_id: str, doc_id: str) -> bool:
        """File a document into a container. Idempotent.

        Returns False only if the container does not exist.
        """
        with self._lock:
            container = self._containers.get(container_id)
            if container is None:
                return False
            if doc_id not in container.member_doc_ids:
                container.member_doc_ids.append(doc_id)
                self._doc_index.setdefault(doc_id, set()).add(container_id)
                self._save()
            return True

    def remove_document(self, container_id: str, doc_id: str) -> bool:
        """Remove a document from one container. Returns False if not a member."""
        with self._lock:
            container = self._containers.get(container_id)
            if container is None or doc_id not in container.member_doc_ids:
                return False
            container.member_doc_ids.remove(doc_id)
            self._detach(doc_id, container_id)
            self._save()
            return True

    def set_document_containers(self, doc_id: str, container_ids: list[str]) -> None:
        """Replace a document's whole set of memberships in one call.

        Unknown container ids are ignored. Passing an empty list removes the
        document from every container.
        """
        with self._lock:
            target = {cid for cid in container_ids if cid in self._containers}
            current = set(self._doc_index.get(doc_id, set()))
            for cid in target - current:
                container = self._containers[cid]
                if doc_id not in container.member_doc_ids:
                    container.member_doc_ids.append(doc_id)
            for cid in current - target:
                container = self._containers.get(cid)
                if container is not None and doc_id in container.member_doc_ids:
                    container.member_doc_ids.remove(doc_id)
            if target:
                self._doc_index[doc_id] = target
            else:
                self._doc_index.pop(doc_id, None)
            self._save()

    def forget_document(self, doc_id: str) -> None:
        """Drop a document from every container (for the future doc-delete flow)."""
        with self._lock:
            container_ids = self._doc_index.pop(doc_id, set())
            for cid in container_ids:
                container = self._containers.get(cid)
                if container is not None and doc_id in container.member_doc_ids:
                    container.member_doc_ids.remove(doc_id)
            if container_ids:
                self._save()

    # --- lookups ---------------------------------------------------------
    def containers_for_document(self, doc_id: str) -> list[str]:
        with self._lock:
            return sorted(self._doc_index.get(doc_id, set()))

    def documents_in_container(self, container_id: str) -> list[str]:
        with self._lock:
            container = self._containers.get(container_id)
            return list(container.member_doc_ids) if container else []

    # --- internal --------------------------------------------------------
    def _detach(self, doc_id: str, container_id: str) -> None:
        """Remove one (doc, container) link from the reverse index. Lock held."""
        ids = self._doc_index.get(doc_id)
        if ids is not None:
            ids.discard(container_id)
            if not ids:
                del self._doc_index[doc_id]


_registry: Optional[ContainerRegistry] = None


def get_registry() -> ContainerRegistry:
    """Process-wide singleton (uses settings.working_dir), like get_boost()."""
    global _registry
    if _registry is None:
        _registry = ContainerRegistry()
    return _registry
