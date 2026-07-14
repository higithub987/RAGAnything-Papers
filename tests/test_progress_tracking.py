"""Tests for ProgressTrackingCallback -> task_store progress mapping.

Verifies the reworked bar is a single monotonic overall percentage across the
whole pipeline (parse -> text_insert -> multimodal -> complete) that never
resets and never goes backward, and that events without a real percentage don't
zero it out.
"""

import pytest

pytest.importorskip("lightrag")

from api import task_store  # noqa: E402
from api.rag_manager import ProgressTrackingCallback  # noqa: E402


def _fresh_task(task_id: str, on_disk: str) -> None:
    task_store.delete_task(task_id)
    task_store.create_task(task_id, "doc.pdf", on_disk_name=on_disk)


def _progress(task_id: str):
    return task_store.get_task(task_id).progress


def test_progress_is_monotonic_and_hits_milestones():
    tid, fp = "t-progress-1", "aaaa_doc.pdf"
    _fresh_task(tid, fp)
    cb = ProgressTrackingCallback()

    seq = []

    def record():
        seq.append(_progress(tid))

    cb.on_parse_start(file_path=fp)
    record()
    # A parse event with no real percent (the cloud parser case) must NOT reset
    # or drop the bar -- it just updates the message.
    cb.on_parse_progress(file_path=fp, percent=None, message="uploading")
    record()
    cb.on_parse_complete(file_path=fp)
    record()
    assert _progress(tid) == 20

    cb.on_text_insert_start(file_path=fp)
    record()
    assert _progress(tid) == 22
    cb.on_text_insert_complete(file_path=fp)
    record()
    assert _progress(tid) == 65

    cb.on_multimodal_start(file_path=fp, item_count=2)
    record()
    assert _progress(tid) == 67
    cb.on_multimodal_item_complete(file_path=fp, item_index=1, total_items=2)
    record()
    assert _progress(tid) == pytest.approx(82.0)  # 67 + 0.5*(97-67)
    cb.on_multimodal_item_complete(file_path=fp, item_index=2, total_items=2)
    record()
    assert _progress(tid) == pytest.approx(97.0)
    cb.on_multimodal_complete(file_path=fp)
    record()

    cb.on_document_complete(file_path=fp)
    record()
    assert _progress(tid) == 100

    # Never None after start, and non-decreasing throughout.
    assert all(p is not None for p in seq)
    assert seq == sorted(seq), f"progress went backward: {seq}"


def test_stray_low_percent_does_not_rewind_bar():
    tid, fp = "t-progress-2", "bbbb_doc.pdf"
    _fresh_task(tid, fp)
    cb = ProgressTrackingCallback()

    cb.on_parse_start(file_path=fp)
    cb.on_parse_complete(file_path=fp)  # -> 20
    assert _progress(tid) == 20
    # A late/low parse percentage would map into the 3..20 band (< 20); the
    # monotonic clamp must keep the bar at 20.
    cb.on_parse_progress(file_path=fp, percent=10, message="late line")
    assert _progress(tid) == 20


def test_no_multimodal_items_does_not_crash_or_reset():
    tid, fp = "t-progress-3", "cccc_doc.pdf"
    _fresh_task(tid, fp)
    cb = ProgressTrackingCallback()

    cb.on_text_insert_complete(file_path=fp)  # -> 65
    # total_items == 0 must not raise and must not lower the bar.
    cb.on_multimodal_item_complete(file_path=fp, item_index=0, total_items=0)
    assert _progress(tid) == 65
