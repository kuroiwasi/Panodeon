from __future__ import annotations

import os
from pathlib import Path
import shutil
import site
import sys

PROVIDER_LABELS = {
    "CPUExecutionProvider": "CPU",
    "CUDAExecutionProvider": "CUDA / NVIDIA",
    "DmlExecutionProvider": "DirectML / NVIDIA AMD",
    "ROCMExecutionProvider": "ROCm / AMD Linux",
    "MIGraphXExecutionProvider": "MIGraphX / AMD Linux",
}
SELECTABLE_PROVIDERS = tuple(PROVIDER_LABELS)
_DLL_DIRECTORY_HANDLES: list[object] = []


def available_onnx_providers() -> list[str]:
    try:
        import onnxruntime as ort
    except ImportError:
        return []
    return list(ort.get_available_providers())


def selectable_onnx_providers(available: list[str] | None = None) -> list[str]:
    available = available_onnx_providers() if available is None else available
    return [
        provider
        for provider in available
        if provider in SELECTABLE_PROVIDERS and provider_dependencies_available(provider)
    ]


def provider_dependencies_available(provider_name: str) -> bool:
    if provider_name != "CUDAExecutionProvider":
        return True
    if preload_cuda_dlls():
        return True
    return all(shutil.which(name) is not None for name in ("cublasLt64_12.dll", "cudnn64_9.dll"))


def preload_dlls_for_providers(providers: list[str]) -> None:
    if "CUDAExecutionProvider" in providers:
        preload_cuda_dlls()


def preload_cuda_dlls() -> bool:
    try:
        import onnxruntime as ort
    except ImportError:
        return False
    add_nvidia_dll_directories()
    preload = getattr(ort, "preload_dlls", None)
    if preload is None:
        return False
    try:
        preload(cuda=True, cudnn=True, msvc=True, directory=None)
        preload(cuda=True, cudnn=True, msvc=False, directory="")
    except Exception:
        return False
    return True


def add_nvidia_dll_directories() -> list[Path]:
    added: list[Path] = []
    add_dll_directory = getattr(os, "add_dll_directory", None)
    if add_dll_directory is None:
        return added
    for directory in nvidia_dll_directories():
        try:
            handle = add_dll_directory(str(directory))
        except OSError:
            continue
        _DLL_DIRECTORY_HANDLES.append(handle)
        added.append(directory)
    return added


def nvidia_dll_directories() -> list[Path]:
    roots = [Path(path) for path in site.getsitepackages()]
    roots.extend(Path(path) for path in sys.path if path)
    directories: list[Path] = []
    seen: set[Path] = set()
    for root in roots:
        nvidia_root = root / "nvidia"
        if not nvidia_root.exists():
            continue
        for directory in nvidia_root.glob("*\\bin"):
            resolved = directory.resolve()
            if resolved not in seen:
                seen.add(resolved)
                directories.append(resolved)
    return directories


def provider_label(provider_name: str | None) -> str:
    if provider_name is None:
        return "No provider"
    return PROVIDER_LABELS.get(provider_name, provider_name)


def session_provider_options(providers: list[str]) -> list[dict[str, str]]:
    options: list[dict[str, str]] = []
    for provider in providers:
        if provider == "CUDAExecutionProvider":
            options.append(
                {
                    "cudnn_conv_algo_search": "EXHAUSTIVE",
                    "use_tf32": "0",
                }
            )
        else:
            options.append({})
    return options


def resolve_execution_providers(provider_name: str | None, available: list[str] | None = None) -> list[str]:
    selectable = selectable_onnx_providers(available)
    if provider_name is None or provider_name not in SELECTABLE_PROVIDERS:
        raise RuntimeError("No ONNX Runtime execution providers are available")
    if provider_name not in selectable:
        label = provider_label(provider_name)
        installed = ", ".join(available_onnx_providers() if available is None else available) or "none"
        raise RuntimeError(f"{label} provider is not available. Installed providers: {installed}")
    return [provider_name, "CPUExecutionProvider"] if provider_name != "CPUExecutionProvider" else [provider_name]
