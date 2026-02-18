#!/usr/bin/env python3
"""Quick environment check for this repository."""

from __future__ import annotations

import importlib
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
]


def check_import(module_name: str, package_hint: str | None) -> tuple[bool, str]:
    try:
        module = importlib.import_module(module_name)
        origin = getattr(module, "__file__", "<built-in>")
        return True, str(origin)
    except Exception as exc:  # pragma: no cover - diagnostics path
        hint = package_hint or module_name
        return False, f"{exc} (try: python -m pip install {hint})"


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

    try:
        import torch  # type: ignore

        print(f"[INFO] torch.cuda.is_available(): {torch.cuda.is_available()}")
    except Exception:
        pass

    try:
        import cellpose  # type: ignore

        resolved = Path(cellpose.__file__).resolve()
        print(f"[INFO] cellpose import path: {resolved}")
        if "Program/cellpose/cellpose" not in str(resolved).replace("\\", "/"):
            print("[WARN] cellpose is not being imported from this repo's local fork.")
    except Exception:
        pass

    print("\nAll required checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
