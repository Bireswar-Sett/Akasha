from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def test_browser_ui_does_not_require_or_construct_a_manifest():
    source = (ROOT / "frontend/src/components/ChatInterface.jsx").read_text()
    assert "manifestJson" not in source
    assert "Input Manifest JSON" not in source
    assert "manifest:" not in source


def test_space_keeps_manifest_json_as_hidden_transport_parameter():
    source = (ROOT / "agent/app.py").read_text()
    assert "def analyze(user_request: str, url_1: str = \"\", url_2: str = \"\", url_3: str = \"\", url_4: str = \"\", manifest_json: str = \"\"" in source
    assert "gr.Textbox(visible=False)" in source
    assert "Input Manifest JSON" not in source
