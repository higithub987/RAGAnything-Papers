import json
from pathlib import Path

from .config import settings


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
