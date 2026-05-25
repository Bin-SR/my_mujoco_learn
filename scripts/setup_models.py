#!/usr/bin/env python3
"""
setup_models.py — Copy Franka Emika Panda assets from pip-installed
mujoco_menagerie into the local models/ directory.

Usage:
    python3 setup_models.py

This is helpful in CI or Docker environments where you want a self-contained
package without relying on runtime path resolution.
"""
import shutil
import sys
from pathlib import Path

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
PANDA_DIR = MODELS_DIR / "franka_emika_panda"


def main():
    # Try to find mujoco_menagerie
    try:
        import mujoco_menagerie
        menagerie_root = Path(mujoco_menagerie.__file__).parent
        source = menagerie_root / "franka_emika_panda"
    except ImportError:
        print("[ERROR] mujoco_menagerie is not installed.")
        print("  Install with:  pip install mujoco-menagerie")
        sys.exit(1)

    if not source.exists():
        print(f"[ERROR] Panda model not found at {source}")
        sys.exit(1)

    if PANDA_DIR.exists():
        print(f"[INFO] Removing existing {PANDA_DIR}")
        shutil.rmtree(PANDA_DIR)

    print(f"[INFO] Copying {source}  ->  {PANDA_DIR}")
    shutil.copytree(source, PANDA_DIR)

    # Verify
    scene = PANDA_DIR / "scene.xml"
    panda = PANDA_DIR / "panda.xml"
    ok_scene = scene.exists()
    ok_panda = panda.exists()

    print(f"  scene.xml : {'OK' if ok_scene else 'MISSING'}")
    print(f"  panda.xml : {'OK' if ok_panda else 'MISSING'}")

    if ok_scene:
        print(f"\n[DONE] Local models are ready at: {MODELS_DIR}")
    else:
        print("\n[WARN] Copy completed but scene.xml not found.")

if __name__ == "__main__":
    main()
