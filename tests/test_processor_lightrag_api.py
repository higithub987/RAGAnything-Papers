import pytest

from raganything.base import DocStatus
from raganything.processor import ProcessorMixin


class FakeLogger:
    def info(self, *args, **kwargs):
        pass

    def warning(self, *args, **kwargs):
        pass

    def error(self, *args, **kwargs):
        pass

    def debug(self, *args, **kwargs):
        pass


class FakeDocStatusStorage:
    def __init__(self):
        self.records = {}
        self.index_done_calls = 0

    async def get_by_id(self, key):
        return self.records.get(key)

    async def upsert(self, data):
        self.records.update(data)

    async def index_done_callback(self):
        self.index_done_calls += 1


class FakeDeletionResult:
    def __init__(self, status="success", message="ok"):
        self.status = status
        self.message = message


@pytest.mark.asyncio
async def test_lightrag_api_init_failure_persists_failed_doc_status():
    class DummyProcessor(ProcessorMixin):
        pass

    processor = DummyProcessor()
    processor.logger = FakeLogger()
    processor.config = type(
        "Config",
        (),
        {
            "use_full_path": False,
            "parser": "mineru",
        },
    )()
    processor.lightrag = type(
        "FakeLightRAG",
        (),
        {"doc_status": FakeDocStatusStorage()},
    )()

    async def fake_ensure_lightrag_initialized():
        return {"success": False, "error": "missing llm_model_func"}

    processor._ensure_lightrag_initialized = fake_ensure_lightrag_initialized

    result = await processor.process_document_complete_lightrag_api("sample.pdf")

    assert result is False
    doc_status = processor.lightrag.doc_status.records["doc-pre-sample.pdf"]
    assert doc_status["status"] == DocStatus.FAILED
    assert doc_status["error_msg"] == "missing llm_model_func"
    assert doc_status["file_path"] == "sample.pdf"
    assert processor.lightrag.doc_status.index_done_calls == 1


@pytest.mark.asyncio
async def test_rollback_document_deletes_doc_id_and_marks_pre_id_failed():
    class DummyProcessor(ProcessorMixin):
        pass

    processor = DummyProcessor()
    processor.logger = FakeLogger()

    delete_calls = []

    class FakeLightRAG:
        def __init__(self):
            self.doc_status = FakeDocStatusStorage()

        async def adelete_by_doc_id(self, doc_id):
            delete_calls.append(doc_id)
            return FakeDeletionResult(status="success")

    processor.lightrag = FakeLightRAG()
    processor.lightrag.doc_status.records["doc-pre-sample.pdf"] = {
        "status": DocStatus.HANDLING,
        "file_path": "sample.pdf",
    }

    await processor._rollback_document(
        "doc-real-id",
        doc_pre_id="doc-pre-sample.pdf",
        error_msg="insertion exploded",
    )

    assert delete_calls == ["doc-real-id"]
    doc_status = processor.lightrag.doc_status.records["doc-pre-sample.pdf"]
    assert doc_status["status"] == DocStatus.FAILED
    assert doc_status["error_msg"] == "insertion exploded"
    assert doc_status["file_path"] == "sample.pdf"


@pytest.mark.asyncio
async def test_rollback_document_skips_delete_when_doc_id_is_none():
    class DummyProcessor(ProcessorMixin):
        pass

    processor = DummyProcessor()
    processor.logger = FakeLogger()

    delete_calls = []

    class FakeLightRAG:
        def __init__(self):
            self.doc_status = FakeDocStatusStorage()

        async def adelete_by_doc_id(self, doc_id):
            delete_calls.append(doc_id)
            return FakeDeletionResult(status="success")

    processor.lightrag = FakeLightRAG()

    await processor._rollback_document(
        None, doc_pre_id="doc-pre-sample.pdf", error_msg="parse failed"
    )

    assert delete_calls == []
    doc_status = processor.lightrag.doc_status.records["doc-pre-sample.pdf"]
    assert doc_status["status"] == DocStatus.FAILED
    assert doc_status["error_msg"] == "parse failed"
