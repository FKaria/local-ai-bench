"""Zero-dependency deterministic single-player game environments for multi-turn benchmarking.

Designed around a minimal reset()/step()/status() interface so TextArena environments
(`ta.make(...)`) can also be plugged in later as another backend in `make_game_env`.
"""

from __future__ import annotations

import random
import re
from collections import deque


class GameEnv:
    """Base interface for single-player stateful benchmark games."""

    def reset(self) -> str:
        raise NotImplementedError

    def step(self, action: str) -> tuple[str, bool, dict]:
        raise NotImplementedError

    def status(self) -> dict:
        raise NotImplementedError


# ---------------------------------------------------------------------------
# 1. Mastermind
# ---------------------------------------------------------------------------


class MastermindEnv(GameEnv):
    """Deduce a hidden sequence of digits from Exact / Misplaced feedback."""

    def __init__(
        self,
        code_len: int = 5,
        num_colors: int = 8,
        max_turns: int = 12,
        seed: int = 42,
    ) -> None:
        self.code_len = code_len
        self.num_colors = num_colors
        self.max_turns = max_turns
        self.seed = seed
        self._secret: list[int] = []
        self.history: list[tuple[list[int], int, int]] = []
        self.turns = 0
        self.invalid_moves = 0
        self.best_exact = 0
        self.won = False
        self.done = False
        self.reset()

    def reset(self) -> str:
        rng = random.Random(self.seed)
        self._secret = [rng.randint(1, self.num_colors) for _ in range(self.code_len)]
        self.history = []
        self.turns = 0
        self.invalid_moves = 0
        self.best_exact = 0
        self.won = False
        self.done = False
        return self._render()

    def _render(self, notice: str = "") -> str:
        lines = [
            f"=== MASTERMIND (length={self.code_len}, digits 1..{self.num_colors}, duplicates allowed) ===",
            f"Turn: {self.turns}/{self.max_turns} | Status: {'WON' if self.won else ('LOST' if self.done else 'IN PROGRESS')}",
        ]
        if notice:
            lines.append(f"Notice: {notice}")
        if self.history:
            lines.append("History:")
            for idx, (guess, exact, misplaced) in enumerate(self.history, 1):
                g_str = " ".join(str(x) for x in guess)
                lines.append(f"  #{idx:2d}: [{g_str}] -> Exact={exact}, Misplaced={misplaced}")
        else:
            lines.append("History: (no guesses yet)")
        if not self.done:
            lines.append(f"Command: guess <{self.code_len} digits from 1..{self.num_colors} separated by spaces>")
        return "\n".join(lines)

    def step(self, action: str) -> tuple[str, bool, dict]:
        if self.done:
            return self._render("Game is already over."), True, self.status()

        cleaned = action.strip()
        cleaned = re.sub(r"^\[|\]$", "", cleaned).strip()
        cleaned = re.sub(r"^(?:guess|move)\s+", "", cleaned, flags=re.IGNORECASE).strip()

        tokens = re.findall(r"\d+", cleaned)
        # Also allow compact digit string like "12345" when code_len <= 9
        if len(tokens) == 1 and len(tokens[0]) == self.code_len and self.num_colors <= 9:
            digits = [int(ch) for ch in tokens[0]]
        else:
            digits = [int(t) for t in tokens]

        if len(digits) != self.code_len or any(not (1 <= d <= self.num_colors) for d in digits):
            self.invalid_moves += 1
            return (
                self._render(
                    f"Invalid guess {action!r}. Provide exactly {self.code_len} integers between 1 and {self.num_colors}."
                ),
                self.done,
                self.status(),
            )

        self.turns += 1
        exact = sum(a == b for a, b in zip(digits, self._secret))
        # Count color overlaps minus exact matches
        rem_secret: dict[int, int] = {}
        rem_guess: dict[int, int] = {}
        for s_val, g_val in zip(self._secret, digits):
            if s_val != g_val:
                rem_secret[s_val] = rem_secret.get(s_val, 0) + 1
                rem_guess[g_val] = rem_guess.get(g_val, 0) + 1
        misplaced = sum(min(rem_secret.get(k, 0), v) for k, v in rem_guess.items())

        self.history.append((digits, exact, misplaced))
        self.best_exact = max(self.best_exact, exact)

        if exact == self.code_len:
            self.won = True
            self.done = True
            return self._render(f"CRACKED IT in {self.turns} guesses!"), True, self.status()

        if self.turns >= self.max_turns:
            self.done = True
            return self._render("Out of guesses. Game over."), True, self.status()

        return self._render(f"Guess recorded: Exact={exact}, Misplaced={misplaced}"), False, self.status()

    def status(self) -> dict:
        progress = 1.0 if self.won else round(self.best_exact / self.code_len, 3)
        return {
            "game": "mastermind",
            "won": self.won,
            "done": self.done,
            "turns": self.turns,
            "max_turns": self.max_turns,
            "invalid_moves": self.invalid_moves,
            "progress": progress,
            "summary": (
                f"Won in {self.turns}/{self.max_turns} guesses"
                if self.won
                else f"Not solved after {self.turns}/{self.max_turns} guesses (best_exact={self.best_exact}/{self.code_len})"
            ),
        }


# ---------------------------------------------------------------------------
# 2. Minesweeper
# ---------------------------------------------------------------------------


class MinesweeperEnv(GameEnv):
    """Single-player deterministic Minesweeper with guaranteed safe opening."""

    def __init__(
        self,
        rows: int = 7,
        cols: int = 7,
        num_mines: int = 8,
        max_turns: int = 60,
        seed: int = 1337,
        start_cell: tuple[int, int] = (3, 3),
    ) -> None:
        self.rows = rows
        self.cols = cols
        self.num_mines = num_mines
        self.max_turns = max_turns
        self.seed = seed
        self.start_cell = start_cell
        self._mines: set[tuple[int, int]] = set()
        self.revealed: set[tuple[int, int]] = set()
        self.flagged: set[tuple[int, int]] = set()
        self.turns = 0
        self.invalid_moves = 0
        self.hit_mine: tuple[int, int] | None = None
        self.won = False
        self.done = False
        self.reset()

    def _adj_mines(self, r: int, c: int) -> int:
        return sum(
            (r + dr, c + dc) in self._mines
            for dr in (-1, 0, 1)
            for dc in (-1, 0, 1)
            if dr or dc
        )

    def _flood_reveal(self, sr: int, sc: int) -> int:
        newly = 0
        q = deque([(sr, sc)])
        while q:
            r, c = q.popleft()
            if (r, c) in self.revealed:
                continue
            self.revealed.add((r, c))
            self.flagged.discard((r, c))
            newly += 1
            if self._adj_mines(r, c) == 0:
                for dr in (-1, 0, 1):
                    for dc in (-1, 0, 1):
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < self.rows and 0 <= nc < self.cols:
                            if (nr, nc) not in self.revealed and (nr, nc) not in self._mines:
                                q.append((nr, nc))
        return newly

    def reset(self) -> str:
        rng = random.Random(self.seed)
        sr, sc = self.start_cell
        forbidden = {(sr + dr, sc + dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1)}
        candidates = [
            (r, c)
            for r in range(self.rows)
            for c in range(self.cols)
            if (r, c) not in forbidden
        ]
        self._mines = set(rng.sample(candidates, self.num_mines))
        self.revealed = set()
        self.flagged = set()
        self.turns = 0
        self.invalid_moves = 0
        self.hit_mine = None
        self.won = False
        self.done = False
        self._flood_reveal(sr, sc)
        return self._render(f"Initial safe opening revealed around ({sr}, {sc}).")

    def _render(self, notice: str = "") -> str:
        total_safe = self.rows * self.cols - self.num_mines
        lines = [
            f"=== MINESWEEPER ({self.rows}x{self.cols}, {self.num_mines} mines) ===",
            f"Safe cells revealed: {len(self.revealed)}/{total_safe} | Flags: {len(self.flagged)} | Turns: {self.turns}/{self.max_turns} | Status: {'WON' if self.won else ('LOST' if self.done else 'IN PROGRESS')}",
        ]
        if notice:
            lines.append(f"Notice: {notice}")
        header = "    " + " ".join(str(c) for c in range(self.cols))
        lines.append(header)
        lines.append("   " + "-" * (2 * self.cols + 1))
        for r in range(self.rows):
            row_chars = []
            for c in range(self.cols):
                if self.hit_mine == (r, c):
                    row_chars.append("*")
                elif (r, c) in self.revealed:
                    row_chars.append(str(self._adj_mines(r, c)))
                elif (r, c) in self.flagged:
                    row_chars.append("F")
                else:
                    row_chars.append(".")
            lines.append(f"{r} | " + " ".join(row_chars))
        lines.append("Legend: . = hidden, F = flagged, 0-8 = adjacent mines")
        if not self.done:
            lines.append("Commands: `reveal <row> <col>` | `flag <row> <col>` (0-indexed)")
        return "\n".join(lines)

    def step(self, action: str) -> tuple[str, bool, dict]:
        if self.done:
            return self._render("Game is already over."), True, self.status()

        cleaned = re.sub(r"^\[|\]$", "", action.strip()).strip()
        m = re.match(r"^(reveal|r|flag|f|unflag)\s+(\d+)[\s,]+(\d+)$", cleaned, flags=re.IGNORECASE)
        if not m:
            self.invalid_moves += 1
            return (
                self._render(f"Invalid command {action!r}. Use `reveal <row> <col>` or `flag <row> <col>`."),
                self.done,
                self.status(),
            )

        verb = m.group(1).lower()
        r, c = int(m.group(2)), int(m.group(3))
        if not (0 <= r < self.rows and 0 <= c < self.cols):
            self.invalid_moves += 1
            return (
                self._render(f"Coordinates ({r}, {c}) out of bounds (0..{self.rows-1}, 0..{self.cols-1})."),
                self.done,
                self.status(),
            )

        self.turns += 1
        if verb in ("flag", "f", "unflag"):
            if (r, c) in self.revealed:
                self.invalid_moves += 1
                msg = f"Cell ({r}, {c}) is already revealed."
            elif (r, c) in self.flagged:
                self.flagged.remove((r, c))
                msg = f"Removed flag at ({r}, {c})."
            else:
                self.flagged.add((r, c))
                msg = f"Placed flag at ({r}, {c})."
        else:
            if (r, c) in self.revealed:
                self.invalid_moves += 1
                msg = f"Cell ({r}, {c}) was already revealed."
            elif (r, c) in self._mines:
                self.hit_mine = (r, c)
                self.done = True
                return self._render(f"BOOM! Hit a mine at ({r}, {c}). Game over."), True, self.status()
            else:
                newly = self._flood_reveal(r, c)
                msg = f"Revealed ({r}, {c}) (+{newly} cell(s))."

        total_safe = self.rows * self.cols - self.num_mines
        if len(self.revealed) >= total_safe:
            self.won = True
            self.done = True
            return self._render(f"ALL {total_safe} SAFE CELLS REVEALED! YOU WIN in {self.turns} turns!"), True, self.status()

        if self.turns >= self.max_turns:
            self.done = True
            return self._render("Reached max turns without clearing all safe cells."), True, self.status()

        return self._render(msg), False, self.status()

    def status(self) -> dict:
        total_safe = self.rows * self.cols - self.num_mines
        progress = 1.0 if self.won else round(len(self.revealed) / total_safe, 3)
        if self.won:
            summary = f"Won in {self.turns}/{self.max_turns} turns"
        elif self.hit_mine is not None:
            summary = f"Hit mine at {self.hit_mine} ({len(self.revealed)}/{total_safe} safe cells revealed)"
        else:
            summary = f"Incomplete ({len(self.revealed)}/{total_safe} safe cells revealed, {self.turns}/{self.max_turns} turns)"
        return {
            "game": "minesweeper",
            "won": self.won,
            "done": self.done,
            "turns": self.turns,
            "max_turns": self.max_turns,
            "invalid_moves": self.invalid_moves,
            "progress": progress,
            "summary": summary,
        }


# ---------------------------------------------------------------------------
# 3. Sokoban
# ---------------------------------------------------------------------------


DEFAULT_SOKOBAN_LEVEL = [
    "#######",
    "#  .  #",
    "# $.$ #",
    "#  @  #",
    "# #$# #",
    "#  .  #",
    "#######",
]


class SokobanEnv(GameEnv):
    """Single-player Sokoban puzzle with move & reset commands."""

    DIRS = {
        "up": (-1, 0),
        "u": (-1, 0),
        "down": (1, 0),
        "d": (1, 0),
        "left": (0, -1),
        "l": (0, -1),
        "right": (0, 1),
        "r": (0, 1),
    }

    def __init__(self, level: list[str] | None = None, max_turns: int = 80) -> None:
        self.level = list(level or DEFAULT_SOKOBAN_LEVEL)
        self.max_turns = max_turns
        self.rows = len(self.level)
        self.cols = max(len(r) for r in self.level)
        self.walls: set[tuple[int, int]] = set()
        self.goals: set[tuple[int, int]] = set()
        self.boxes: set[tuple[int, int]] = set()
        self.player: tuple[int, int] = (0, 0)
        self.turns = 0
        self.resets = 0
        self.invalid_moves = 0
        self.won = False
        self.done = False
        self.reset()

    def _load_board(self) -> None:
        self.walls = set()
        self.goals = set()
        self.boxes = set()
        for r, row in enumerate(self.level):
            for c, ch in enumerate(row):
                if ch == "#":
                    self.walls.add((r, c))
                elif ch == ".":
                    self.goals.add((r, c))
                elif ch == "$":
                    self.boxes.add((r, c))
                elif ch == "*":
                    self.boxes.add((r, c))
                    self.goals.add((r, c))
                elif ch == "@":
                    self.player = (r, c)
                elif ch == "+":
                    self.player = (r, c)
                    self.goals.add((r, c))

    def reset(self) -> str:
        self._load_board()
        self.turns = 0
        self.resets = 0
        self.invalid_moves = 0
        self.won = False
        self.done = False
        return self._render("Push all boxes ($) onto target goals (.).")

    def _render(self, notice: str = "") -> str:
        on_goal = len(self.boxes & self.goals)
        lines = [
            f"=== SOKOBAN ({on_goal}/{len(self.goals)} boxes on goals) ===",
            f"Moves: {self.turns}/{self.max_turns} | Resets: {self.resets} | Status: {'WON' if self.won else ('LOST' if self.done else 'IN PROGRESS')}",
        ]
        if notice:
            lines.append(f"Notice: {notice}")
        for r in range(self.rows):
            row_chars = []
            for c in range(self.cols):
                pos = (r, c)
                if pos in self.walls:
                    row_chars.append("#")
                elif pos == self.player:
                    row_chars.append("+" if pos in self.goals else "@")
                elif pos in self.boxes:
                    row_chars.append("*" if pos in self.goals else "$")
                elif pos in self.goals:
                    row_chars.append(".")
                else:
                    row_chars.append(" ")
            lines.append("".join(row_chars))
        lines.append("Legend: # wall, @ player, + player on goal, $ box, . goal, * box on goal")
        if not self.done:
            lines.append("Commands: `up` | `down` | `left` | `right` | `reset` (restarts board, keeps move counter)")
        return "\n".join(lines)

    def step(self, action: str) -> tuple[str, bool, dict]:
        if self.done:
            return self._render("Game is already over."), True, self.status()

        cleaned = re.sub(r"^\[|\]$", "", action.strip()).strip().lower()
        cleaned = re.sub(r"^move\s+", "", cleaned).strip()

        if cleaned == "reset":
            self.resets += 1
            self.turns += 1
            self._load_board()
            if self.turns >= self.max_turns:
                self.done = True
                return self._render("Out of moves."), True, self.status()
            return self._render("Board reset to initial layout."), False, self.status()

        if cleaned not in self.DIRS:
            self.invalid_moves += 1
            return (
                self._render(f"Invalid move {action!r}. Use one of: up, down, left, right, reset."),
                self.done,
                self.status(),
            )

        dr, dc = self.DIRS[cleaned]
        pr, pc = self.player
        nr, nc = pr + dr, pc + dc

        self.turns += 1
        if (nr, nc) in self.walls:
            self.invalid_moves += 1
            msg = f"Blocked by wall moving {cleaned}."
        elif (nr, nc) in self.boxes:
            br, bc = nr + dr, nc + dc
            if (br, bc) in self.walls or (br, bc) in self.boxes:
                self.invalid_moves += 1
                msg = f"Cannot push box {cleaned} (blocked)."
            else:
                self.boxes.remove((nr, nc))
                self.boxes.add((br, bc))
                self.player = (nr, nc)
                msg = f"Pushed box {cleaned}."
        else:
            self.player = (nr, nc)
            msg = f"Moved {cleaned}."

        if self.boxes == self.goals:
            self.won = True
            self.done = True
            return self._render(f"PUZZLE SOLVED in {self.turns} moves!"), True, self.status()

        if self.turns >= self.max_turns:
            self.done = True
            return self._render("Out of moves. Game over."), True, self.status()

        return self._render(msg), False, self.status()

    def status(self) -> dict:
        on_goal = len(self.boxes & self.goals)
        progress = 1.0 if self.won else round(on_goal / max(1, len(self.goals)), 3)
        return {
            "game": "sokoban",
            "won": self.won,
            "done": self.done,
            "turns": self.turns,
            "max_turns": self.max_turns,
            "invalid_moves": self.invalid_moves,
            "progress": progress,
            "summary": (
                f"Solved in {self.turns}/{self.max_turns} moves (resets={self.resets})"
                if self.won
                else f"Not solved ({on_goal}/{len(self.goals)} boxes on goals, {self.turns}/{self.max_turns} moves)"
            ),
        }


def extract_game_action(text: str, game_type: str) -> str:
    """Extract a game move from a model's text reply (prefers bracketed `[action]`, falls back to \\boxed{})."""
    gtype = game_type.lower()
    bracketed = [m.group(1).strip() for m in re.finditer(r"\[([^\[\]\n]{1,64})\]", text)]

    if gtype == "mastermind":
        # Prefer last bracketed guess containing digits
        for cand in reversed(bracketed):
            if re.search(r"\d", cand):
                return cand
    elif gtype == "minesweeper":
        for cand in reversed(bracketed):
            if re.match(r"^(?:reveal|r|flag|f|unflag)\s+\d+[\s,]+\d+$", cand, flags=re.IGNORECASE):
                return cand
    elif gtype == "sokoban":
        for cand in reversed(bracketed):
            if re.match(r"^(?:move\s+)?(?:up|down|left|right|u|d|l|r|reset)$", cand, flags=re.IGNORECASE):
                return cand

    if bracketed:
        return bracketed[-1]

    m_box = re.findall(r"\\boxed\{([^}\n]+)\}", text)
    if m_box:
        return m_box[-1].strip()

    # Last non-empty line fallback
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    return lines[-1][:64] if lines else ""


def make_game_env(task: dict) -> GameEnv:
    """Instantiate a GameEnv from a task dictionary."""
    gtype = str(task["game"]).lower()
    cfg = dict(task.get("game_config") or {})
    if gtype == "mastermind":
        return MastermindEnv(**cfg)
    if gtype == "minesweeper":
        if "start_cell" in cfg and isinstance(cfg["start_cell"], list):
            cfg["start_cell"] = tuple(cfg["start_cell"])
        return MinesweeperEnv(**cfg)
    if gtype == "sokoban":
        return SokobanEnv(**cfg)
    raise ValueError(f"Unknown game type: {gtype}")
