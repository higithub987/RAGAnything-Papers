# RAG-Anything

A RAG-Anything personal project by Leo Jiang, assisted by Claude Code, used for querying a
database of academic papers and other files.

Stack: [RAG-Anything](https://github.com/HKUDS/RAG-Anything) + LightRAG for retrieval, Milvus
for vector storage, and MinerU (cloud trial API by default) for document parsing. The API
service lives in [`api/`](api/) and is served with FastAPI/uvicorn.

## Setup

```bash
uv sync                       # create .venv and install pinned deps from uv.lock
```

Then create `.env` in the repo root (it is git-ignored — never commit it) with at least:

- `LLM_BINDING_API_KEY`, `EMBEDDING_BINDING_API_KEY` — your model provider keys.
- `MINERU_API_TOKEN` — get one at https://mineru.net (API management). Used by
  `PARSER=mineru-api` (the default; cloud parsing, no local GPU).
- Milvus must be reachable at `MILVUS_URI` (default `http://localhost:19530`).

Run the API:

```bash
uv run uvicorn api.main:app --host 0.0.0.0 --port 9621
```

## Required post-install patch — atomic `doc_status` writes

> Apply this once after every `uv sync` / fresh venv, or `GET /documents` will intermittently
> 500 while a document is indexing.

**Why this is needed.** LightRAG's `write_json` (in `lightrag/utils.py`) writes
non-atomically: it truncates the target file to 0 bytes and *then* streams JSON into it. The
API polls `kv_store_doc_status.json` on every `GET /documents`, and during indexing LightRAG
rewrites that file constantly — so a poll landing in the write window reads an empty/partial
file and crashes with `json.JSONDecodeError: Expecting value: line 1 column 1 (char 0)`.

The fix makes the write atomic (serialize to a temp file in the same directory, then
`os.replace()` onto the target), so readers always see a complete file. It lives in
`lightrag/utils.py` **inside `.venv/`**, which is git-ignored — so it does **not** travel with
the repo and must be re-applied whenever the environment is rebuilt. (The API also has
in-code read guards in `api/rag_manager.py` that keep the endpoint from 500-ing even without
this patch; this patch removes the race at the source for *all* readers of LightRAG's JSON
stores.)

**Apply it** (idempotent — safe to run repeatedly; run from the repo root):

```bash
uv run python - <<'PY'
import pathlib, lightrag.utils as u
p = pathlib.Path(u.__file__)
src = p.read_text(encoding="utf-8")

if "Write atomically:" in src:
    print("Already patched:", p); raise SystemExit(0)

# Ensure `import tempfile` is present (stock has: import os / import re / import time)
if "import tempfile" not in src:
    src = src.replace("import re\nimport time\n", "import re\nimport tempfile\nimport time\n", 1)

old = '''    try:
        # Strategy 1: Fast path - try direct serialization
        with open(file_name, "w", encoding="utf-8") as f:
            json.dump(json_obj, f, indent=2, ensure_ascii=False)
        return False  # No sanitization needed, no reload required

    except (UnicodeEncodeError, UnicodeDecodeError) as e:
        logger.debug(f"Direct JSON write failed, using sanitizing encoder: {e}")

    # Strategy 2: Use custom encoder (sanitizes during serialization, zero memory copy)
    with open(file_name, "w", encoding="utf-8") as f:
        json.dump(json_obj, f, indent=2, ensure_ascii=False, cls=SanitizingJSONEncoder)

    logger.info(f"JSON sanitization applied during write: {file_name}")
    return True  # Sanitization applied, reload recommended'''

new = '''    # Write atomically: temp file in the same dir, then os.replace() onto the
    # target, so concurrent readers never see a truncated/empty file (fixes the
    # doc_status read race that 500s GET /documents during indexing).
    directory = os.path.dirname(os.path.abspath(file_name))
    fd, tmp_path = tempfile.mkstemp(dir=directory, prefix=".tmp-", suffix=".json")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            try:
                json.dump(json_obj, f, indent=2, ensure_ascii=False)
                sanitized = False
            except (UnicodeEncodeError, UnicodeDecodeError) as e:
                logger.debug(f"Direct JSON write failed, using sanitizing encoder: {e}")
                f.seek(0)
                f.truncate()
                json.dump(json_obj, f, indent=2, ensure_ascii=False, cls=SanitizingJSONEncoder)
                sanitized = True
        os.replace(tmp_path, file_name)
    except BaseException:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise
    if sanitized:
        logger.info(f"JSON sanitization applied during write: {file_name}")
    return sanitized'''

if old not in src:
    raise SystemExit(
        "write_json body not found — the installed LightRAG version differs from the one "
        "this patch targets. Patch lightrag/utils.py:write_json to write atomically by hand."
    )

p.write_text(src.replace(old, new), encoding="utf-8")
print("Patched:", p)
PY
```

**Verify** the patch took (should print `True`):

```bash
uv run python -c "import inspect, lightrag.utils as u; print('os.replace' in inspect.getsource(u.write_json))"
```

**Durable alternative.** Because this edits a file under `.venv/`, it is lost on every
reinstall. If you'd rather it survive automatically, apply the same rewrite at startup from
tracked code (e.g. rebind `write_json` to an atomic version at the top of
`initialize_rag()` in `api/rag_manager.py`), or maintain it as an upstream LightRAG PR.

## Tests

```bash
uv run pytest tests/ -q
```
