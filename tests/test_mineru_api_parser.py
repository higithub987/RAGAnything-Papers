"""Tests for the MinerU online-trial cloud API parser (``MineruApiParser``).

All tests are fully offline: the three HTTP steps and the result-zip download are
mocked, so no token or network access is required (CI-safe).
"""

import io
import json
import zipfile

import pytest

from raganything.parser import (
    MineruApiParser,
    MineruParser,
    get_parser,
    get_supported_parsers,
    list_parsers,
)


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _make_zip(members: dict) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        for name, data in members.items():
            zf.writestr(name, data)
    return buf.getvalue()


class _FakeResp:
    """Minimal context-manager stand-in for an http response."""

    def __init__(self, blob: bytes, status: int = 200):
        self._blob = blob
        self.status = status

    def read(self):
        return self._blob

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def _stub_happy_flow(monkeypatch, parser, zip_bytes, *, data_id_echo=True):
    """Wire a parser instance so upload/poll/download return a canned success."""
    captured = {"put": None, "posted": None}

    def fake_post(url, body):
        captured["posted"] = (url, body)
        return {
            "code": 0,
            "msg": "ok",
            "data": {"batch_id": "batch-1", "file_urls": ["https://oss/upload"]},
        }

    def fake_put(url, data):
        captured["put"] = (url, len(data))

    def fake_get(url):
        did = captured["posted"][1]["files"][0]["data_id"] if data_id_echo else "x"
        return {
            "code": 0,
            "msg": "ok",
            "data": {
                "extract_result": [
                    {
                        "data_id": did,
                        "file_name": "f.pdf",
                        "state": "done",
                        "err_msg": "",
                        "full_zip_url": "https://cdn/result.zip",
                    }
                ]
            },
        }

    monkeypatch.setattr(parser, "_post_json", fake_post)
    monkeypatch.setattr(parser, "_put_file", fake_put)
    monkeypatch.setattr(parser, "_get_json", fake_get)
    # Only the zip download goes through urlopen once the helpers are stubbed.
    import raganything.parser as parser_mod

    monkeypatch.setattr(
        parser_mod.urllib.request,
        "urlopen",
        lambda url, timeout=None, context=None: _FakeResp(zip_bytes),
    )
    return captured


# --------------------------------------------------------------------------- #
# wiring / registry
# --------------------------------------------------------------------------- #
def test_get_parser_returns_api_parser():
    assert isinstance(get_parser("mineru-api"), MineruApiParser)
    assert isinstance(get_parser("mineru-cloud"), MineruApiParser)


def test_api_parser_is_a_mineru_parser():
    # So the processor's isinstance(..., MineruParser) progress gate still applies.
    assert isinstance(MineruApiParser(), MineruParser)


def test_registry_lists_api_parser():
    assert "mineru-api" in get_supported_parsers()
    assert list_parsers().get("mineru-api") == "MineruApiParser"


def test_check_installation_requires_token(monkeypatch):
    monkeypatch.delenv("MINERU_API_TOKEN", raising=False)
    assert MineruApiParser().check_installation() is False
    monkeypatch.setenv("MINERU_API_TOKEN", "tok-123")
    assert MineruApiParser().check_installation() is True


# --------------------------------------------------------------------------- #
# end-to-end (mocked) happy path
# --------------------------------------------------------------------------- #
def test_parse_pdf_end_to_end_mocked(monkeypatch, tmp_path):
    monkeypatch.setenv("MINERU_API_TOKEN", "tok-123")
    parser = MineruApiParser()

    content = [
        {"type": "text", "text": "hello world", "page_idx": 0},
        {
            "type": "image",
            "img_path": "images/pic.jpg",
            "image_caption": ["a caption"],
            "page_idx": 1,
        },
    ]
    zip_bytes = _make_zip(
        {
            "abc123_content_list.json": json.dumps(content),
            "abc123_content_list_v2.json": json.dumps([{"junk": True}]),  # must be ignored
            "full.md": "# hello",
            "images/pic.jpg": b"\xff\xd8\xff\xe0fake",
        }
    )
    _stub_happy_flow(monkeypatch, parser, zip_bytes)

    pdf = tmp_path / "sample.pdf"
    pdf.write_bytes(b"%PDF-1.4\n%stub\n")

    out = tmp_path / "out"
    result = parser.parse_pdf(str(pdf), output_dir=str(out))

    assert [it["type"] for it in result] == ["text", "image"]
    img = next(it for it in result if it["type"] == "image")
    # _read_output_files should have made the path absolute and it must exist.
    from pathlib import Path

    assert Path(img["img_path"]).is_absolute()
    assert Path(img["img_path"]).exists()
    assert img["img_path"].endswith("pic.jpg")


def test_parse_pdf_request_maps_options(monkeypatch, tmp_path):
    monkeypatch.setenv("MINERU_API_TOKEN", "tok-123")
    parser = MineruApiParser()
    zip_bytes = _make_zip(
        {"x_content_list.json": json.dumps([{"type": "text", "text": "t", "page_idx": 0}])}
    )
    captured = _stub_happy_flow(monkeypatch, parser, zip_bytes)

    pdf = tmp_path / "s.pdf"
    pdf.write_bytes(b"%PDF-1.4\n")
    parser.parse_pdf(
        str(pdf),
        output_dir=str(tmp_path / "o"),
        method="ocr",
        lang="en",
        formula=False,
        table=False,
        start_page=1,
        end_page=3,
        # MinerU-local kwargs that must be tolerated (ignored), not raise:
        device="cuda",
        backend="pipeline",
        vlm_url="http://x",
    )
    _, body = captured["posted"]
    assert body["is_ocr"] is True
    assert body["enable_formula"] is False
    assert body["enable_table"] is False
    assert body["language"] == "en"
    assert body["page_ranges"] == "1-3"


# --------------------------------------------------------------------------- #
# failure / safety
# --------------------------------------------------------------------------- #
def test_failed_state_raises(monkeypatch, tmp_path):
    monkeypatch.setenv("MINERU_API_TOKEN", "tok-123")
    parser = MineruApiParser()

    monkeypatch.setattr(
        parser,
        "_post_json",
        lambda url, body: {
            "code": 0,
            "data": {"batch_id": "b", "file_urls": ["https://oss/u"]},
        },
    )
    monkeypatch.setattr(parser, "_put_file", lambda url, data: None)
    monkeypatch.setattr(
        parser,
        "_get_json",
        lambda url: {
            "code": 0,
            "data": {
                "extract_result": [
                    {"data_id": "x", "state": "failed", "err_msg": "bad pdf"}
                ]
            },
        },
    )
    pdf = tmp_path / "s.pdf"
    pdf.write_bytes(b"%PDF-1.4\n")
    with pytest.raises(RuntimeError, match="bad pdf"):
        parser.parse_pdf(str(pdf), output_dir=str(tmp_path / "o"))


def test_api_error_envelope_raises(monkeypatch, tmp_path):
    monkeypatch.setenv("MINERU_API_TOKEN", "tok-123")
    parser = MineruApiParser()
    monkeypatch.setattr(
        parser, "_post_json", lambda url, body: {"code": 401, "msg": "unauthorized"}
    )
    pdf = tmp_path / "s.pdf"
    pdf.write_bytes(b"%PDF-1.4\n")
    with pytest.raises(RuntimeError, match="unauthorized"):
        parser.parse_pdf(str(pdf), output_dir=str(tmp_path / "o"))


def test_missing_content_list_in_zip_raises(monkeypatch, tmp_path):
    monkeypatch.setenv("MINERU_API_TOKEN", "tok-123")
    parser = MineruApiParser()
    zip_bytes = _make_zip({"full.md": "# only markdown, no content list"})
    _stub_happy_flow(monkeypatch, parser, zip_bytes)
    pdf = tmp_path / "s.pdf"
    pdf.write_bytes(b"%PDF-1.4\n")
    with pytest.raises(RuntimeError, match="content_list"):
        parser.parse_pdf(str(pdf), output_dir=str(tmp_path / "o"))


def test_no_upload_when_token_missing(monkeypatch, tmp_path):
    monkeypatch.delenv("MINERU_API_TOKEN", raising=False)
    parser = MineruApiParser()
    called = {"n": 0}
    monkeypatch.setattr(
        parser, "_post_json", lambda *a, **k: called.__setitem__("n", called["n"] + 1)
    )
    pdf = tmp_path / "s.pdf"
    pdf.write_bytes(b"%PDF-1.4\n")
    with pytest.raises(RuntimeError, match="MINERU_API_TOKEN"):
        parser.parse_pdf(str(pdf), output_dir=str(tmp_path / "o"))
    assert called["n"] == 0


# --------------------------------------------------------------------------- #
# input validation (all local, no network)
# --------------------------------------------------------------------------- #
def test_validate_rejects_unsupported_extension(tmp_path):
    parser = MineruApiParser()
    bad = tmp_path / "malware.exe"
    bad.write_bytes(b"MZ\x90\x00")
    with pytest.raises(ValueError, match="cannot upload"):
        parser._validate_upload(bad)


def test_validate_rejects_extension_content_mismatch(tmp_path):
    parser = MineruApiParser()
    fake_pdf = tmp_path / "notreally.pdf"
    fake_pdf.write_bytes(b"<html>error page</html>")
    with pytest.raises(ValueError, match="does not match"):
        parser._validate_upload(fake_pdf)


def test_validate_rejects_empty_file(tmp_path):
    parser = MineruApiParser()
    empty = tmp_path / "empty.pdf"
    empty.write_bytes(b"")
    with pytest.raises(ValueError, match="empty"):
        parser._validate_upload(empty)


def test_validate_rejects_oversize_file(tmp_path):
    parser = MineruApiParser()
    parser.MAX_UPLOAD_BYTES = 4  # shrink limit instead of writing 200 MB
    pdf = tmp_path / "big.pdf"
    pdf.write_bytes(b"%PDF-1.4 and then some more bytes")
    with pytest.raises(ValueError, match="exceeds"):
        parser._validate_upload(pdf)


# --------------------------------------------------------------------------- #
# archive safety (zip-slip)
# --------------------------------------------------------------------------- #
def test_safe_extract_rejects_zip_slip(tmp_path):
    dest = tmp_path / "dest"
    dest.mkdir()
    evil = _make_zip({"../evil.txt": b"pwned", "ok.txt": b"fine"})
    with pytest.raises(RuntimeError, match="escapes target"):
        MineruApiParser._safe_extract_zip(evil, dest)
    # Nothing must have been written outside the destination dir.
    assert not (tmp_path / "evil.txt").exists()


def test_safe_extract_writes_normal_members(tmp_path):
    dest = tmp_path / "dest"
    dest.mkdir()
    good = _make_zip({"a.txt": b"1", "sub/b.txt": b"2"})
    MineruApiParser._safe_extract_zip(good, dest)
    assert (dest / "a.txt").read_bytes() == b"1"
    assert (dest / "sub" / "b.txt").read_bytes() == b"2"
