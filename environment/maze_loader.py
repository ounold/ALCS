from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Tuple


@dataclass(frozen=True)
class ACS2MazeDefinition:
    name: str
    rows: int
    cols: int
    start_pos: Tuple[int, int]
    goal_pos: Tuple[int, int]
    obstacles: Tuple[Tuple[int, int], ...]

    def to_parameters(self) -> Dict[str, object]:
        return {
            "rows": self.rows,
            "cols": self.cols,
            "start_pos": list(self.start_pos),
            "goal_pos": list(self.goal_pos),
            "obstacles": [list(item) for item in self.obstacles],
            "maze_source": "acs2_upstream",
            "maze_name": self.name,
        }


ACS2_MAZE_ROOT = Path(__file__).resolve().parent / "acs2_mazes"

_TOKEN_MAP = {
    "OBSTACLE": 1,
    "CORRIDOR": 0,
    "PRIZE": 2,
}


def _extract_int(pattern: str, source: str, label: str) -> int:
    match = re.search(pattern, source)
    if not match:
        raise ValueError(f"Could not parse {label} from maze source")
    return int(match.group(1))


def _extract_goal(source: str) -> Tuple[int, int]:
    match = re.search(r"\.goalState\s*=\s*\{\s*(\d+)\s*,\s*(\d+)\s*\}", source)
    if not match:
        raise ValueError("Could not parse goalState from maze source")
    return int(match.group(1)), int(match.group(2))


def _extract_rows(source: str) -> Iterable[Tuple[str, ...]]:
    rows = re.findall(r"\{([^{}]+)\}", source)
    cleaned_rows = []
    for row in rows:
        tokens = [token.strip() for token in row.split(",") if token.strip()]
        if tokens and all(token in _TOKEN_MAP for token in tokens):
            cleaned_rows.append(tuple(tokens))
    return cleaned_rows


def parse_acs2_maze_file(path: Path) -> ACS2MazeDefinition:
    source = path.read_text(encoding="utf-8")
    rows = _extract_int(r"\.mazeHeight\s*=\s*(\d+)", source, "mazeHeight")
    cols = _extract_int(r"\.mazeWidth\s*=\s*(\d+)", source, "mazeWidth")
    goal_pos = _extract_goal(source)
    grid_rows = list(_extract_rows(source))
    if len(grid_rows) != rows:
        raise ValueError(f"Expected {rows} maze rows in {path.name}, found {len(grid_rows)}")
    for row in grid_rows:
        if len(row) != cols:
            raise ValueError(f"Expected {cols} columns in {path.name}, found {len(row)}")

    obstacles = []
    corridors = []
    for row_idx, row in enumerate(grid_rows):
        for col_idx, token in enumerate(row):
            if token == "OBSTACLE":
                obstacles.append((row_idx, col_idx))
            else:
                corridors.append((row_idx, col_idx))

    start_candidates = [pos for pos in corridors if pos != goal_pos]
    if not start_candidates:
        raise ValueError(f"No valid start positions found in {path.name}")

    return ACS2MazeDefinition(
        name=path.stem,
        rows=rows,
        cols=cols,
        start_pos=min(start_candidates),
        goal_pos=goal_pos,
        obstacles=tuple(obstacles),
    )


def load_acs2_maze_catalog(root: Path | None = None) -> Dict[str, ACS2MazeDefinition]:
    maze_root = root or ACS2_MAZE_ROOT
    if not maze_root.exists():
        return {}
    catalog: Dict[str, ACS2MazeDefinition] = {}
    for path in sorted(maze_root.glob("*.cpp")):
        definition = parse_acs2_maze_file(path)
        catalog[definition.name] = definition
    return catalog
