from __future__ import annotations

import sys
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

RUN_ROOT = Path(__file__).resolve().parents[1] / "data" / "ingest_runs"


class TeeStream:
    def __init__(self, stream, log_handle):
        self.stream = stream
        self.log_handle = log_handle

    def write(self, data: str) -> int:
        written = self.stream.write(data)
        self.log_handle.write(data)
        self.flush()
        return written

    def flush(self) -> None:
        self.stream.flush()
        self.log_handle.flush()

    def isatty(self) -> bool:
        return bool(getattr(self.stream, "isatty", lambda: False)())

    @property
    def encoding(self) -> str | None:
        return getattr(self.stream, "encoding", None)


def resolve_run_dir(categories: list[str]) -> Path:
    normalized = [item.strip() for item in categories if item.strip()]
    if len(normalized) == 1:
        scope = normalized[0]
    elif normalized:
        scope = "mixed"
    else:
        scope = "manual"
    path = RUN_ROOT / scope
    path.mkdir(parents=True, exist_ok=True)
    return path


@contextmanager
def capture_run_output(run_dir: Path, *, argv: list[str]):
    run_log = run_dir / "run.log"
    with run_log.open("a", encoding="utf-8") as handle:
        stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        handle.write(f"\n[{stamp}] $ {' '.join(argv)}\n")
        handle.flush()

        original_stdout = sys.stdout
        original_stderr = sys.stderr
        sys.stdout = TeeStream(original_stdout, handle)
        sys.stderr = TeeStream(original_stderr, handle)
        try:
            yield run_log
        finally:
            sys.stdout = original_stdout
            sys.stderr = original_stderr
