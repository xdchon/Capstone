#!/usr/bin/env python3
"""Quick environment check for this repository."""

from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SBREAD_DIR = ROOT / "SBReadFile22-Python-main"
if str(SBREAD_DIR) not in sys.path:
    sys.path.insert(0, str(SBREAD_DIR))

REQUIRED_IMPORTS = [
    ("numpy", None),
    ("tifffile", None),
    ("yaml", "pyyaml"),
    ("pyzstd", None),
    ("torch", None),
    ("cellpose", None),
    ("SBReadFile", None),
    ("PyQt6", None),
    ("qtpy", None),
    ("pyqtgraph", None),
    ("superqt", None),
]


def check_import(module_name: str, package_hint: str | None) -> tuple[bool, str]:
    try:
        module = importlib.import_module(module_name)
        origin = getattr(module, "__file__", None)
        if module_name == "cellpose" and not origin:
            return (
                False,
                "cellpose resolved as a namespace package without __file__ "
                "(install local fork with: python -m pip install -e ./Program/cellpose)",
            )
        if origin is None:
            origin = "<built-in>"
        return True, str(origin)
    except Exception as exc:  # pragma: no cover - diagnostics path
        hint = package_hint or module_name
        return False, f"{exc} (try: python -m pip install {hint})"


def _find_qt_windows_plugin() -> tuple[bool, str]:
    """Return whether qwindows.dll is discoverable, plus diagnostics."""
    candidates: list[Path] = []

    env_platform = os.environ.get("QT_QPA_PLATFORM_PLUGIN_PATH", "").strip()
    env_plugin = os.environ.get("QT_PLUGIN_PATH", "").strip()
    conda_prefix = os.environ.get("CONDA_PREFIX", "").strip()

    if env_platform:
        candidates.append(Path(env_platform))
    if env_plugin:
        candidates.append(Path(env_plugin) / "platforms")
    if conda_prefix:
        candidates.append(Path(conda_prefix) / "Library" / "plugins" / "platforms")
        candidates.append(Path(conda_prefix) / "Lib" / "site-packages" / "PyQt6" / "Qt6" / "plugins" / "platforms")
        candidates.append(Path(conda_prefix) / "lib" / "site-packages" / "PyQt6" / "Qt6" / "plugins" / "platforms")
        candidates.append(Path(conda_prefix) / "Lib" / "site-packages" / "PySide6" / "plugins" / "platforms")
        candidates.append(Path(conda_prefix) / "lib" / "site-packages" / "PySide6" / "plugins" / "platforms")

    try:
        from qtpy.QtCore import QLibraryInfo  # type: ignore

        qt_plugins = QLibraryInfo.path(QLibraryInfo.LibraryPath.PluginsPath)
        if qt_plugins:
            candidates.append(Path(qt_plugins) / "platforms")
            candidates.append(Path(qt_plugins))
    except Exception:
        pass

    seen: set[str] = set()
    unique_candidates: list[Path] = []
    for c in candidates:
        key = str(c.resolve()) if c.exists() else str(c)
        if key not in seen:
            seen.add(key)
            unique_candidates.append(c)

    for c in unique_candidates:
        if c.exists():
            matches = sorted(c.glob("qwindows*.dll"))
            if matches:
                return True, f"{matches[0]}"

    diag = [
        f"QT_QPA_PLATFORM_PLUGIN_PATH={env_platform or '<unset>'}",
        f"QT_PLUGIN_PATH={env_plugin or '<unset>'}",
        f"CONDA_PREFIX={conda_prefix or '<unset>'}",
        "searched=" + "; ".join(str(c) for c in unique_candidates) if unique_candidates else "searched=<none>",
    ]
    return False, " | ".join(diag)


def main() -> int:
    failures = 0

    print("Checking required imports...")
    for module_name, package_hint in REQUIRED_IMPORTS:
        ok, detail = check_import(module_name, package_hint)
        if ok:
            print(f"[OK]   {module_name}: {detail}")
        else:
            failures += 1
            print(f"[FAIL] {module_name}: {detail}")

    if failures:
        print(f"\n{failures} check(s) failed.")
        return 1

    # Verify that qtpy can resolve an actual Qt binding.
    try:
        from qtpy import QtWidgets  # type: ignore

        _ = QtWidgets.QApplication
        print("[OK]   qtpy QtWidgets binding resolved")
    except Exception as exc:
        print(f"[FAIL] qtpy QtWidgets binding: {exc}")
        print("[HINT] In PowerShell, try: Remove-Item Env:\\QT_API; $env:QT_API='pyqt6'")
        return 1

    if sys.platform.startswith("win"):
        ok, detail = _find_qt_windows_plugin()
        if ok:
            print(f"[OK]   Qt Windows plugin found: {detail}")
        else:
            print(f"[FAIL] Qt Windows plugin not found: {detail}")
            print(
                "[HINT] In PowerShell, set "
                "$env:QT_PLUGIN_PATH=\"$env:CONDA_PREFIX\\Library\\plugins\" and "
                "$env:QT_QPA_PLATFORM_PLUGIN_PATH=\"$env:CONDA_PREFIX\\Library\\plugins\\platforms\""
            )
            return 1

    try:
        import torch  # type: ignore

        print(f"[INFO] torch.cuda.is_available(): {torch.cuda.is_available()}")
    except Exception:
        pass

    try:
        import cellpose  # type: ignore

        cellpose_file = getattr(cellpose, "__file__", None)
        if cellpose_file:
            resolved = Path(cellpose_file).resolve()
            print(f"[INFO] cellpose import path: {resolved}")
            if "Program/cellpose/cellpose" not in str(resolved).replace("\\", "/"):
                print("[WARN] cellpose is not being imported from this repo's local fork.")
        else:
            print(
                "[WARN] cellpose import did not resolve to a concrete file path "
                "(namespace package)."
            )
    except Exception:
        pass

    print("\nAll required checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
