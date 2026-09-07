import ast
from pathlib import Path


SERVER_PATH = Path(__file__).parents[1] / "inference" / "server.py"


def test_gradio_geochat_uses_filepath_and_preserves_api_name():
    source = SERVER_PATH.read_text()
    tree = ast.parse(source)

    assert "gr.Image" not in source
    assert 'gr.File(type="filepath"' in source
    assert 'api_name="geochat"' in source
    assert "@app.post(\"/http/geochat/sar\")" in source


def test_server_compiles_without_loading_model_weights():
    compile(SERVER_PATH.read_text(), str(SERVER_PATH), "exec")
