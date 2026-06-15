#!/usr/bin/env python3
from __future__ import annotations

import sys


def main() -> None:
    print(
        "ingest_recent_arxiv.py has been split into two scripts:\n"
        "  1. python scripts/download_recent_arxiv.py ...\n"
        "  2. python scripts/ingest_local_arxiv.py ...\n",
        file=sys.stderr,
    )
    raise SystemExit(1)


if __name__ == "__main__":
    main()
