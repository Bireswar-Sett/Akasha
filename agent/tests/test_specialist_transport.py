from pathlib import Path

from qwen.controller.services import HuggingFaceService


class FakeResponse:
    def __init__(self, payload=None, status_code=200, lines=None):
        self.payload = payload
        self.status_code = status_code
        self._lines = lines or []

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")

    def json(self):
        return self.payload

    def iter_lines(self, decode_unicode=True):
        return iter(self._lines)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


def test_http_transport_uploads_submits_and_polls(tmp_path, monkeypatch):
    image = tmp_path / "sample.tif"
    image.write_bytes(b"tiff-data")
    calls = []

    def fake_post(url, **kwargs):
        calls.append(("POST", url, kwargs))
        if url.endswith("/upload"):
            return FakeResponse(["/tmp/uploaded/sample.tif"])
        return FakeResponse({"event_id": "event-123"})

    def fake_get(url, **kwargs):
        calls.append(("GET", url, kwargs))
        return FakeResponse(lines=["event: complete", 'data: [{"answer": "ok"}]'])

    monkeypatch.setattr("qwen.controller.services.requests.post", fake_post)
    monkeypatch.setattr("qwen.controller.services.requests.get", fake_get)

    service = HuggingFaceService("Bireswar26/GeoChat", "/geochat", token="test-token")
    result = service.predict({"image": str(image), "prompt": "Describe", "max_new_tokens": 128})

    assert result == [{"answer": "ok"}]
    assert calls[0][1].endswith("/gradio_api/upload")
    assert calls[1][1].endswith("/gradio_api/call/v2/geochat")
    assert calls[2][1].endswith("/gradio_api/call/geochat/event-123")
    assert calls[1][2]["json"]["image"]["meta"]["_type"] == "gradio.FileData"
    assert calls[1][2]["json"]["prompt"] == "Describe"
    assert calls[1][2]["json"]["max_new_tokens"] == 128


def test_space_url_uses_live_hugging_face_hostname():
    assert HuggingFaceService._space_url("Bireswar26/TeoChat") == "https://bireswar26-teochat.hf.space"
