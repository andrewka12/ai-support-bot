"""ai_support_bot -- a customer-support bot grounded in a business knowledge base.

The package answers customer questions strictly from documents in ``kb/``,
refuses to invent answers, and hands complex or sensitive cases to a human
operator with a tracked ticket.

Entry points:
    ``python -m ai_support_bot chat``   -- offline console demo (no token)
    ``python -m ai_support_bot ingest`` -- build the retrieval index
    ``python -m ai_support_bot stats``  -- analytics over the dialogue log
    ``python -m ai_support_bot serve``  -- real Telegram bot (long polling)
"""

from __future__ import annotations

__version__ = "1.0.0"
__all__ = ["__version__"]
