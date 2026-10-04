"""Test package for ai_support_bot.

Importing this package puts the project root on ``sys.path`` so the tests can
be started both as ``python -m unittest discover -s tests`` and as
``python tests/test_retrieval.py`` from any working directory.
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
