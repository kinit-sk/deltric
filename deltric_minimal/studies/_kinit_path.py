"""Shared sys.path shim for scripts in this folder that reuse kinit's
edge-feature engine (``edge_pruning_study.py``) instead of duplicating it.

These studies are DelTriC-specific (they exist to inform DelTriC's edge
pruning), but the feature-extraction machinery they build on is generic and
lives in the sibling ``kinit`` repo, alongside the dim-reduction-comparison
scripts that also depend on it. Rather than fork that engine, this repo is
expected to be checked out next to ``kinit`` (both under the same parent
directory), and importers just need this shim first::

    import _kinit_path  # noqa: F401  (adds ../../../kinit to sys.path)
    import edge_pruning_study as eps

Set the ``KINIT_REPO`` environment variable to override the sibling-directory
assumption.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

_default = Path(__file__).resolve().parents[3] / "kinit"
KINIT_DIR = Path(os.environ.get("KINIT_REPO", _default))

if not (KINIT_DIR / "edge_pruning_study.py").exists():
    raise ModuleNotFoundError(
        f"kinit repo not found at {KINIT_DIR}. Check it out next to this repo, "
        "or set KINIT_REPO to its path."
    )

if str(KINIT_DIR) not in sys.path:
    sys.path.insert(0, str(KINIT_DIR))
