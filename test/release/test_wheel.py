"""Evidence 39: a fresh wheel owns its Python entry point and Rust codec."""

from pathlib import Path
import subprocess
import sys
from zipfile import ZipFile


def test_isolated_wheel_imports_and_executes_rust_codec(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[2]
    subprocess.run(
        [sys.executable, "-m", "maturin", "build", "--offline", "--out", str(tmp_path)],
        cwd=root, check=True, capture_output=True, text=True,
    )
    wheel, = tmp_path.glob("tllm-*.whl")
    with ZipFile(wheel) as archive:
        assert not any(
            name.startswith(("tllm/runtime/", "tllm/ports/", "tllm/common/"))
            for name in archive.namelist()
        )
    installed = tmp_path / "installed"
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "--no-deps", "--no-index",
         "--target", str(installed), str(wheel)],
        cwd=tmp_path, check=True, capture_output=True, text=True,
    )
    result = subprocess.run(
        [sys.executable, "-I", "-c", """
import importlib.metadata
import json
from pathlib import Path
import sys

installed = Path(sys.argv[1]).resolve()
sys.path.insert(0, str(installed))
import tllm
from tllm import _core

assert Path(tllm.__file__).resolve().is_relative_to(installed)
assert Path(_core.__file__).resolve().is_relative_to(installed)
distribution = importlib.metadata.Distribution.at(installed / sys.argv[2])
assert not any(e.group == 'vllm.general_plugins' for e in distribution.entry_points)
request = json.dumps({'schema_version': 1, 'layouts': [{
    'resource': 'wheel', 'layout': {'namespace': 'wheel', 'schema_version': 1,
    'fields': [{'field_id': 0, 'shape': [2, 4], 'dtype': 'float32',
    'alignment_bytes': 16, 'mutation_granularity': 'row'}]}}]})
decision = _core.intern_canonical_tensor_layouts(request)
_core.validate_canonical_tensor_layouts(request, decision)
assert json.loads(decision)['bindings'][0]['layout_id'] == 0
assert callable(tllm.make_llm)
print('isolated wheel: Rust layout codec executed')
""", str(installed), next(installed.glob("tllm-*.dist-info")).name],
        cwd=tmp_path, check=True, capture_output=True, text=True,
    )
    assert "Rust layout codec executed" in result.stdout
