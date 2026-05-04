from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
PYTHON = ROOT / ".venv" / "Scripts" / "python.exe"
CONFIG = ROOT / "experiments" / "configs" / "batch_mazes.yaml"

MAZES = [
    "Cassandra4x4",
    "Littman57",
    "Littman89",
    "Maze10",
    "Maze4",
    "Maze7",
    "MazeA",
    "MazeB",
    "MazeD",
    "MazeE1",
    "MazeE2",
    "MazeE3",
    "MazeF1",
    "MazeF2",
    "MazeF3",
    "MazeF4",
    "MazeMA",
    "MiyazakiA",
    "MiyazakiB",
    "Woods1",
    "Woods100",
    "Woods101",
    "Woods101_5",
    "Woods102",
]

MODES = ["cpu_single", "cpu_mp", "gpu", "gpu_seq"]
DEVICE_BY_MODE = {
    "cpu_single": "cpu",
    "cpu_mp": "cpu",
    "gpu": "cuda",
    "gpu_seq": "cuda",
}


def main() -> int:
    if not PYTHON.exists():
        print(f"Python executable not found: {PYTHON}", file=sys.stderr)
        return 1
    if not CONFIG.exists():
        print(f"Config file not found: {CONFIG}", file=sys.stderr)
        return 1

    print(f'Running ACS2 dashboards for all mazes and modes using "{CONFIG}"')
    print("no_subsumption=true will be forced on every run.")
    print()

    total_runs = len(MAZES) * len(MODES)
    current = 0

    for maze in MAZES:
        for mode in MODES:
            current += 1
            device = DEVICE_BY_MODE[mode]
            cmd = [
                str(PYTHON),
                "acs2.py",
                "--config",
                str(CONFIG),
                "--environment_name",
                maze,
                "--explore_mode",
                mode,
                "--exploit_mode",
                mode,
                "--device",
                device,
                "--no_subsumption",
                "true",
            ]

            print("=" * 68)
            print(f"[{current}/{total_runs}] Maze: {maze} | Mode: {mode} | Device: {device}")
            print("=" * 68)
            completed = subprocess.run(cmd, cwd=ROOT)
            if completed.returncode != 0:
                print(f"FAILED: maze={maze} mode={mode} device={device}", file=sys.stderr)
                return completed.returncode
            print()

    print("All runs completed successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
