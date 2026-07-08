import json
import re
from pathlib import Path

from .config import settings

# Mirrors web/documents.html's UPLOAD_PREFIX_RE: uploads are stored on disk as
# "<task_uuid>_<original name>", so a doc name derived from the on-disk path
# carries that prefix. Strip it so internal names match the documents tab.
_UPLOAD_PREFIX_RE = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}_",
    re.IGNORECASE,
)


def strip_upload_prefix(file_name: str) -> str:
    return _UPLOAD_PREFIX_RE.sub("", file_name or "")


# Same UUID prefix as _UPLOAD_PREFIX_RE but matched anywhere in a blob of text,
# for scrubbing it out of the query context/citations sent to the LLM (where it
# appears mid-text, once per reference line). The lookbehind keeps it from biting
# into a longer hex run.
_UPLOAD_PREFIX_ANYWHERE_RE = re.compile(
    r"(?<![0-9a-f])[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}_",
    re.IGNORECASE,
)


def strip_upload_prefixes(text: str) -> str:
    return _UPLOAD_PREFIX_ANYWHERE_RE.sub("", text or "")


def _path() -> Path:
    return Path(settings.working_dir) / "doc_original_names.json"


def save_doc_name(doc_id: str, file_name: str) -> None:
    p = _path()
    data = json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}
    data[doc_id] = file_name
    p.write_text(json.dumps(data), encoding="utf-8")


def load_doc_names() -> dict[str, str]:
    p = _path()
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}
