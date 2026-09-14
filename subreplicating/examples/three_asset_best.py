"""Backward-compatible alias.

The "best-fit" search (RA and BRA, variance and K-aware objectives, Nelder-Mead
polish) is now the default pipeline of ``three_asset.py``; this module is kept
so that ``python -m subreplicating.examples.three_asset_best`` still works.
"""

from __future__ import annotations

from .three_asset import main

if __name__ == "__main__":
    main()
