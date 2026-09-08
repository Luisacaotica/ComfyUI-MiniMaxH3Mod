"""Run upstream and character regressions from a standalone checkout.

Use ComfyUI's Python: python tools/run_tests.py /path/to/ComfyUI
No server, model downloads, or generation jobs are started.
"""
import argparse
import sys
import types
import unittest
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("comfyui", type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    comfy_dir = args.comfyui.resolve()
    if not (comfy_dir / "comfy/cli_args.py").is_file():
        parser.error("Expected a ComfyUI source directory")
    sys.path.insert(0, str(comfy_dir))
    sys.argv = [sys.argv[0]]
    import comfy.cli_args
    comfy.cli_args.args.cpu = True
    # Upstream tests import custom_nodes.<folder>; allow that same package
    # layout without installing/copying this checkout into a live ComfyUI.
    namespace = types.ModuleType("custom_nodes")
    namespace.__path__ = [str(root.parent), str(comfy_dir / "custom_nodes")]
    sys.modules["custom_nodes"] = namespace
    suite = unittest.defaultTestLoader.discover(str(root / "tests"), pattern="test_*.py")
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())
