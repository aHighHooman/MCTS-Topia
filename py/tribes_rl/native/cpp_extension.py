from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
from typing import Optional

from torch.utils.cpp_extension import load


_NATIVE_MCTS_MODULE = None
_MSVC_ENV_READY = False
_BUILD_FLAGS_VERSION = "mcts-opt-v13-strict-parity"


def _candidate_vsdevcmd_paths() -> list[Path]:
    return [
        Path("C:/Program Files/Microsoft Visual Studio/18/Community/Common7/Tools/VsDevCmd.bat"),
        Path("C:/Program Files/Microsoft Visual Studio/17/Community/Common7/Tools/VsDevCmd.bat"),
        Path("C:/Program Files (x86)/Microsoft Visual Studio/2022/Community/Common7/Tools/VsDevCmd.bat"),
    ]


def _ensure_msvc_env() -> bool:
    global _MSVC_ENV_READY
    if _MSVC_ENV_READY:
        return shutil.which("cl") is not None
    if shutil.which("cl") is not None:
        _MSVC_ENV_READY = True
        return True

    vsdevcmd = next((path for path in _candidate_vsdevcmd_paths() if path.exists()), None)
    if vsdevcmd is None:
        return False

    script_path: str | None = None
    try:
        with tempfile.NamedTemporaryFile("w", suffix=".cmd", delete=False, encoding="utf-8") as handle:
            handle.write("@echo off\n")
            handle.write(f'call "{vsdevcmd}" -host_arch=x64 -arch=x64 >nul\n')
            handle.write("set\n")
            script_path = handle.name
        completed = subprocess.run(
            ["cmd.exe", "/d", "/c", script_path],
            capture_output=True,
            text=True,
            check=True,
        )
    except Exception:
        return False
    finally:
        if script_path:
            try:
                os.unlink(script_path)
            except OSError:
                pass

    for line in completed.stdout.splitlines():
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        if key:
            os.environ[key] = value

    _MSVC_ENV_READY = True
    return shutil.which("cl") is not None


def load_native_mcts_extension() -> Optional[object]:
    global _NATIVE_MCTS_MODULE
    if _NATIVE_MCTS_MODULE is not None:
        return _NATIVE_MCTS_MODULE

    source = Path(__file__).with_name("native_mcts.cpp")
    rules_source = source.with_name("native_rules.cpp")
    static_eval_source = source.with_name("native_static_eval.cpp")
    build_dir = source.parent / ".build"
    build_dir.mkdir(parents=True, exist_ok=True)
    stamp = build_dir / "tribes_rl_native_mcts.flags"

    if os.name == "nt":
        extra_cflags = [
            "/O2",
            "/std:c++17",
        ]
        extra_ldflags = []
    else:
        extra_cflags = [
            "-O3",
            "-std=c++17",
        ]
        extra_ldflags = []
    flags_signature = "\n".join([_BUILD_FLAGS_VERSION, *extra_cflags, *extra_ldflags])

    existing_binaries = [
        path
        for path in list(build_dir.glob("tribes_rl_native_mcts*.pyd")) + list(build_dir.glob("tribes_rl_native_mcts*.so"))
        if "debug" not in path.stem
    ]
    existing_binary = next(iter(existing_binaries), None)
    if (
        existing_binary is not None
        and existing_binary.stat().st_mtime >= source.stat().st_mtime
        and existing_binary.stat().st_mtime >= rules_source.stat().st_mtime
        and existing_binary.stat().st_mtime >= static_eval_source.stat().st_mtime
        and stamp.exists()
        and stamp.read_text(encoding="utf-8") == flags_signature
    ):
        spec = importlib.util.spec_from_file_location("tribes_rl_native_mcts", existing_binary)
        if spec is not None and spec.loader is not None:
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            _NATIVE_MCTS_MODULE = module
            return _NATIVE_MCTS_MODULE

    if os.name == "nt" and not _ensure_msvc_env():
        _NATIVE_MCTS_MODULE = None
        return _NATIVE_MCTS_MODULE

    try:
        _NATIVE_MCTS_MODULE = load(
            name="tribes_rl_native_mcts",
            sources=[str(source), str(rules_source), str(static_eval_source)],
            extra_cflags=extra_cflags,
            extra_ldflags=extra_ldflags,
            build_directory=str(build_dir),
            verbose=False,
        )
        stamp.write_text(flags_signature, encoding="utf-8")
    except Exception:
        _NATIVE_MCTS_MODULE = None
    return _NATIVE_MCTS_MODULE
