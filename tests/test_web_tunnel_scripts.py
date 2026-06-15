from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _write_executable(path: Path, contents: str) -> None:
    path.write_text(contents, encoding="utf-8")
    path.chmod(0o755)


def _copy_scripts(tmp_path: Path) -> Path:
    root = tmp_path / "project"
    scripts_dir = root / "scripts"
    scripts_dir.mkdir(parents=True)
    shutil.copy(PROJECT_ROOT / "scripts" / "deploy.sh", scripts_dir / "deploy.sh")
    shutil.copy(PROJECT_ROOT / "scripts" / "stop.sh", scripts_dir / "stop.sh")
    return root


def _fake_path(tmp_path: Path) -> Path:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()

    _write_executable(
        bin_dir / "python3",
        """#!/usr/bin/env bash
set -euo pipefail
if [[ "${1:-}" == "-c" ]]; then
  if [[ "${2:-}" == *"json.load"* ]]; then
    cat >/dev/null
  fi
  exit 0
fi
if [[ "${1:-}" == "-m" && "${2:-}" == "scholar_mind.db.init_db" ]]; then
  exit 0
fi
if [[ "${1:-}" == "-m" && ( "${2:-}" == "uvicorn" || "${2:-}" == "celery" ) ]]; then
  trap 'exit 0' TERM INT
  while true; do /bin/sleep 1; done
fi
exit 0
""",
    )
    _write_executable(
        bin_dir / "curl",
        """#!/usr/bin/env bash
set -euo pipefail
for arg in "$@"; do
  case "$arg" in
    *6333/collections*) echo '{"result":[]}' ; exit 0 ;;
    *api/v1/health*) echo '{"data":{"status":"healthy"}}' ; exit 0 ;;
  esac
done
exit 0
""",
    )
    _write_executable(
        bin_dir / "cloudflared",
        """#!/usr/bin/env bash
set -euo pipefail
if [[ "${1:-}" == "tunnel" ]]; then
  printf 'INF https://script-test.trycloudflare.com\\n' >&2
  trap 'exit 0' TERM INT
  while true; do /bin/sleep 1; done
fi
exit 1
""",
    )
    return bin_dir


def _script_env(root: Path, bin_dir: Path) -> dict[str, str]:
    env = os.environ.copy()
    env.update(
        {
            "PATH": f"{bin_dir}:{env['PATH']}",
            "STARTUP_TIMEOUT_SECONDS": "5",
            "STOP_TIMEOUT_SECONDS": "3",
            "LOG_FILE": str(root / "data" / "logs" / "scholar_mind.log"),
        }
    )
    return env


def _run_stop(root: Path, env: dict[str, str]) -> None:
    subprocess.run(
        ["bash", str(root / "scripts" / "stop.sh"), "--web"],
        cwd=root,
        env=env,
        text=True,
        capture_output=True,
        timeout=15,
        check=False,
    )


def test_deploy_web_starts_quick_tunnel_and_prints_public_url(tmp_path: Path) -> None:
    root = _copy_scripts(tmp_path)
    bin_dir = _fake_path(tmp_path)
    env = _script_env(root, bin_dir)

    try:
        result = subprocess.run(
            ["bash", str(root / "scripts" / "deploy.sh"), "--web"],
            cwd=root,
            env=env,
            text=True,
            capture_output=True,
            timeout=20,
            check=False,
        )

        assert result.returncode == 0, result.stderr
        assert "https://script-test.trycloudflare.com" in result.stdout
        assert (root / "data" / "run" / "cloudflared.pid").is_file()
        assert (root / "data" / "run" / "cloudflared.url").read_text(
            encoding="utf-8"
        ).strip() == "https://script-test.trycloudflare.com"
    finally:
        _run_stop(root, env)


def test_stop_web_stops_cloudflared_pid_and_removes_files(tmp_path: Path) -> None:
    root = _copy_scripts(tmp_path)
    bin_dir = _fake_path(tmp_path)
    env = _script_env(root, bin_dir)
    run_dir = root / "data" / "run"
    run_dir.mkdir(parents=True)

    proc = subprocess.Popen(
        ["bash", "-c", "trap 'exit 0' TERM INT; while true; do sleep 1; done"]
    )
    try:
        (run_dir / "cloudflared.pid").write_text(str(proc.pid), encoding="utf-8")
        (run_dir / "cloudflared.url").write_text(
            "https://script-test.trycloudflare.com\n", encoding="utf-8"
        )

        result = subprocess.run(
            ["bash", str(root / "scripts" / "stop.sh"), "--web"],
            cwd=root,
            env=env,
            text=True,
            capture_output=True,
            timeout=15,
            check=False,
        )

        assert result.returncode == 0, result.stderr
        assert proc.wait(timeout=5) == 0
        assert not (run_dir / "cloudflared.pid").exists()
        assert not (run_dir / "cloudflared.url").exists()
    finally:
        if proc.poll() is None:
            proc.terminate()
            proc.wait(timeout=5)
