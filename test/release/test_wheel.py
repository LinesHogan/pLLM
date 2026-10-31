"""Evidence 39: a fresh wheel owns its Python entry point and Rust codec."""

import subprocess
import sys
import tarfile
from pathlib import Path
from zipfile import ZipFile

import pytest


@pytest.mark.slow
def test_isolated_wheel_imports_and_executes_rust_codec(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[2]
    subprocess.run(
        [sys.executable, "-m", "maturin", "sdist", "--out", str(tmp_path)],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    (source_archive,) = tmp_path.glob("pllm-*.tar.gz")
    source = tmp_path / "source"
    with tarfile.open(source_archive) as archive:
        archive.extractall(source, filter="data")
    (project,) = source.iterdir()
    subprocess.run(
        [sys.executable, "-m", "maturin", "build", "--release", "--offline", "--out", str(tmp_path)],
        cwd=project,
        check=True,
        capture_output=True,
        text=True,
    )
    (wheel,) = tmp_path.glob("pllm-*.whl")
    with ZipFile(wheel) as archive:
        expected_sources = {
            path.relative_to(root).as_posix() for path in (root / "pllm").rglob("*") if path.suffix in (".py", ".pyi")
        }
        packaged_sources = {
            name for name in archive.namelist() if name.startswith("pllm/") and Path(name).suffix in (".py", ".pyi")
        }
        assert packaged_sources == expected_sources
        (license_path,) = (name for name in archive.namelist() if name.endswith("/licenses/LICENSE"))
        assert b"Apache License" in archive.read(license_path)
        assert not any(name.startswith(("pllm/runtime/", "pllm/ports/", "pllm/common/")) for name in archive.namelist())
        assert not set(archive.namelist()).intersection(
            {
                "pllm/backends/vllm/overlay_state.py",
                "pllm/backends/vllm/versioned_parameter_overlay.py",
                "pllm/backends/vllm/dense_parameter_overlay.py",
            }
        )
    installed = tmp_path / "installed"
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "--no-deps", "--no-index", "--target", str(installed), str(wheel)],
        cwd=tmp_path,
        check=True,
        capture_output=True,
        text=True,
    )
    result = subprocess.run(
        [
            sys.executable,
            "-I",
            "-c",
            """
import importlib.metadata
import importlib
import json
from pathlib import Path
import sys
from packaging.requirements import Requirement

installed = Path(sys.argv[1]).resolve()
sys.path.insert(0, str(installed))
import pllm
import pytest
import torch
from pllm import _core

assert Path(pllm.__file__).resolve().is_relative_to(installed)
assert Path(_core.__file__).resolve().is_relative_to(installed)
distribution = importlib.metadata.Distribution.at(installed / sys.argv[2])
assert not any(e.group == 'vllm.general_plugins' for e in distribution.entry_points)
requirements = [Requirement(value) for value in distribution.requires or ()]
assert any(req.name == 'torch' and req.marker is None for req in requirements)
request = json.dumps({'schema_version': 1, 'layouts': [{
    'resource': 'wheel', 'layout': {'namespace': 'wheel', 'schema_version': 1,
    'fields': [{'field_id': 0, 'shape': [2, 4], 'dtype': 'float32',
    'alignment_bytes': 16}]}}]})
decision = _core.intern_tensor_layouts(request)
assert json.loads(decision)['bindings'][0]['layout_id'] == 0
assert callable(pllm.make_llm)
for workload in ('dummy', 'esamp_request', 'fttt', 'gtta', 'inplace_ttt', 'tlm',
                 'reft', 'recirculation', 'override', 'ttrl', 'lora'):
    module = importlib.import_module('pllm.workloads.' + workload)
    assert Path(module.__file__).resolve().is_relative_to(installed)
from pllm.host import HostRuntimeFacts
from pllm.workloads.dummy import DummyWorkload
execution = pllm.make_llm(workload=DummyWorkload(),
    config=HostRuntimeFacts(torch.device('cpu')))
assert not torch.cuda.is_initialized()
print('isolated wheel: Rust layout codec executed')
""",
            str(installed),
            next(installed.glob("pllm-*.dist-info")).name,
        ],
        cwd=tmp_path,
        check=True,
        capture_output=True,
        text=True,
    )
    assert "Rust layout codec executed" in result.stdout
