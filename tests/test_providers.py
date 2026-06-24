from __future__ import annotations

import pytest

from panodeon.inference import providers as provider_module
from panodeon.inference.providers import (
    add_nvidia_dll_directories,
    nvidia_dll_directories,
    provider_dependencies_available,
    provider_label,
    preload_dlls_for_providers,
    resolve_execution_providers,
    selectable_onnx_providers,
    session_provider_options,
)


def test_provider_label_known_provider() -> None:
    assert provider_label("DmlExecutionProvider") == "DirectML / NVIDIA AMD"
    assert provider_label("ROCMExecutionProvider") == "ROCm / AMD Linux"
    assert provider_label("MIGraphXExecutionProvider") == "MIGraphX / AMD Linux"


def test_provider_label_unknown_provider() -> None:
    assert provider_label("AzureExecutionProvider") == "AzureExecutionProvider"


def test_resolve_no_provider() -> None:
    with pytest.raises(RuntimeError, match="No ONNX Runtime execution providers"):
        resolve_execution_providers(None, [])


def test_selectable_providers_excludes_tensorrt(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(provider_module, "provider_dependencies_available", lambda provider: True)
    providers = selectable_onnx_providers(
        ["TensorrtExecutionProvider", "MIGraphXExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"]
    )
    assert providers == ["MIGraphXExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"]


def test_cuda_provider_requires_dlls(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(provider_module, "preload_cuda_dlls", lambda: False)
    monkeypatch.setattr(provider_module.shutil, "which", lambda name: None)
    assert not provider_dependencies_available("CUDAExecutionProvider")


def test_cuda_provider_accepts_preloaded_dlls(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(provider_module, "preload_cuda_dlls", lambda: True)
    assert provider_dependencies_available("CUDAExecutionProvider")


def test_preload_dlls_for_cuda_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = []
    monkeypatch.setattr(provider_module, "preload_cuda_dlls", lambda: calls.append("cuda") or True)
    preload_dlls_for_providers(["CUDAExecutionProvider", "CPUExecutionProvider"])
    assert calls == ["cuda"]


def test_nvidia_dll_directories_finds_package_bin(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    package_bin = tmp_path / "nvidia" / "cudnn" / "bin"
    package_bin.mkdir(parents=True)
    monkeypatch.setattr(provider_module.site, "getsitepackages", lambda: [str(tmp_path)])
    monkeypatch.setattr(provider_module.sys, "path", [])
    assert nvidia_dll_directories() == [package_bin.resolve()]


def test_add_nvidia_dll_directories_keeps_handles(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    package_bin = tmp_path / "nvidia" / "cudnn" / "bin"
    package_bin.mkdir(parents=True)
    handles = []
    monkeypatch.setattr(provider_module, "nvidia_dll_directories", lambda: [package_bin])
    monkeypatch.setattr(provider_module.os, "add_dll_directory", lambda path: handles.append(path) or object())
    assert add_nvidia_dll_directories() == [package_bin]
    assert handles == [str(package_bin)]


def test_cuda_session_provider_options_are_optimized() -> None:
    options = session_provider_options(["CUDAExecutionProvider", "CPUExecutionProvider"])
    assert options[0]["cudnn_conv_algo_search"] == "EXHAUSTIVE"
    assert "cudnn_conv_use_max_workspace" not in options[0]
    assert options[0]["use_tf32"] == "0"
    assert options[1] == {}


def test_resolve_cuda_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(provider_module, "provider_dependencies_available", lambda provider: True)
    providers = resolve_execution_providers(
        "CUDAExecutionProvider",
        ["CPUExecutionProvider", "DmlExecutionProvider", "CUDAExecutionProvider"],
    )
    assert providers == ["CUDAExecutionProvider", "CPUExecutionProvider"]


def test_resolve_directml_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(provider_module, "provider_dependencies_available", lambda provider: True)
    providers = resolve_execution_providers("DmlExecutionProvider", ["CPUExecutionProvider", "DmlExecutionProvider"])
    assert providers == ["DmlExecutionProvider", "CPUExecutionProvider"]


def test_resolve_cpu() -> None:
    assert resolve_execution_providers("CPUExecutionProvider", ["CPUExecutionProvider"]) == ["CPUExecutionProvider"]


def test_resolve_missing_provider_reports_available() -> None:
    with pytest.raises(RuntimeError, match="Installed providers: CPUExecutionProvider"):
        resolve_execution_providers("CUDAExecutionProvider", ["CPUExecutionProvider"])
