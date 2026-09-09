"""api/pagination.py -- shared limit/offset clamping so no endpoint can be
asked to return an unbounded result set (CLAUDE.md Sec 20)."""
from __future__ import annotations

from core.config import ApiConfig


def resolve_pagination(limit: int | None, offset: int | None, api_config: ApiConfig) -> tuple[int, int]:
    resolved_limit = limit if limit is not None else api_config.default_page_size
    resolved_limit = max(1, min(resolved_limit, api_config.max_page_size))
    resolved_offset = max(0, offset or 0)
    return resolved_limit, resolved_offset
