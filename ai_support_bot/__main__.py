"""Allow ``python -m ai_support_bot`` to run the command line interface."""

from __future__ import annotations

from .cli import main

if __name__ == "__main__":
    raise SystemExit(main())
