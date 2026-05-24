"""
_progress.py - Shared progress display utilities for TradingBotV1

Provides:
  _bar(pct, width)  — filled/empty block bar string
  _fmt_time(secs)   — seconds → Xs or MM:SS
  Spinner           — animated spinner for indeterminate tasks
  Step              — context manager for a numbered step with elapsed time
"""

import sys
import time
import threading


# ── Bar characters ─────────────────────────────────────────────────────────────
FILLED  = "█"
EMPTY   = "░"
OK      = "✓"
FAIL    = "✗"
SKIP    = "⊘"
WARN    = "⚠"

SPINNER_FRAMES = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"


def _bar(pct: float, width: int = 20) -> str:
    """Return a block progress bar string for the given percentage (0–100)."""
    pct     = max(0.0, min(100.0, pct))
    filled  = int(round(width * pct / 100))
    return FILLED * filled + EMPTY * (width - filled)


def _fmt_time(secs: float) -> str:
    """Format elapsed seconds as Xs or MM:SS."""
    if secs < 60:
        return f"{secs:.1f}s"
    m, s = divmod(int(secs), 60)
    return f"{m:02d}:{s:02d}"


def _fmt_eta(elapsed: float, pct: float) -> str:
    """Estimate remaining time string given elapsed seconds and % complete."""
    if pct <= 0:
        return "—"
    total_est = elapsed / (pct / 100)
    remaining = total_est - elapsed
    return _fmt_time(remaining)


def print_inline(text: str) -> None:
    """Overwrite the current terminal line."""
    print(f"\r{text}", end="", flush=True)


def print_done() -> None:
    """Move to the next line after an inline print."""
    print()


class Spinner:
    """
    Animated spinner for tasks that cannot show a percentage.

    Usage:
        with Spinner("Fetching news...") as sp:
            result = do_slow_thing()
        # prints: ✓ Fetching news...  (1.3s)

    Or manually:
        sp = Spinner("Connecting to MT5...").start()
        ...
        sp.stop()
    """

    def __init__(self, message: str, indent: int = 2) -> None:
        self.message  = message
        self.indent   = indent
        self._stop    = threading.Event()
        self._thread  = threading.Thread(target=self._run, daemon=True)
        self._start_t = time.time()

    def start(self) -> "Spinner":
        self._start_t = time.time()
        self._thread.start()
        return self

    def _run(self) -> None:
        pad = " " * self.indent
        i   = 0
        while not self._stop.is_set():
            frame   = SPINNER_FRAMES[i % len(SPINNER_FRAMES)]
            elapsed = time.time() - self._start_t
            print_inline(f"{pad}{frame} {self.message}  {_fmt_time(elapsed)}")
            i += 1
            time.sleep(0.1)

    def stop(self, success: bool = True, suffix: str = "") -> float:
        elapsed = time.time() - self._start_t
        self._stop.set()
        self._thread.join()
        pad    = " " * self.indent
        icon   = OK if success else FAIL
        suffix = f"  {suffix}" if suffix else ""
        print(f"\r{pad}{icon} {self.message}{suffix}  ({_fmt_time(elapsed)})")
        return elapsed

    def __enter__(self) -> "Spinner":
        return self.start()

    def __exit__(self, exc_type, *_) -> None:
        self.stop(success=exc_type is None)


class ProgressBar:
    """
    Single-line updating progress bar for counted tasks.

    Usage:
        bar = ProgressBar(total=50, label="Testing rules")
        for item in items:
            bar.update(current_item_name)
        bar.finish()
    """

    def __init__(
        self,
        total:   int,
        label:   str  = "",
        width:   int  = 20,
        indent:  int  = 2,
        show_eta: bool = True,
    ) -> None:
        self.total    = max(1, total)
        self.label    = label
        self.width    = width
        self.indent   = indent
        self.show_eta = show_eta
        self._count   = 0
        self._start   = time.time()

    def update(self, item_label: str = "", count: int = 1) -> None:
        self._count = min(self._count + count, self.total)
        pct     = self._count / self.total * 100
        elapsed = time.time() - self._start
        bar     = _bar(pct, self.width)
        eta_str = f" ETA:{_fmt_eta(elapsed, pct)}" if self.show_eta and pct < 100 else ""
        label_t = f" | {item_label[:40]}" if item_label else ""
        pad     = " " * self.indent
        print_inline(
            f"{pad}{bar} {pct:>4.0f}%  "
            f"{self._count}/{self.total}{label_t}"
            f"  elapsed:{_fmt_time(elapsed)}{eta_str}"
        )

    def finish(self, suffix: str = "") -> float:
        elapsed = time.time() - self._start
        bar     = _bar(100, self.width)
        pad     = " " * self.indent
        extra   = f"  {suffix}" if suffix else ""
        print(
            f"\r{pad}{bar} 100%  "
            f"{self.total}/{self.total}{extra}  "
            f"done ({_fmt_time(elapsed)})"
        )
        return elapsed


class Step:
    """
    Context manager that prints a numbered step header and elapsed time footer.

    Usage:
        with Step(1, 4, "Ingesting resources"):
            do_work()
    """

    def __init__(self, num: int, total: int, title: str) -> None:
        self.num    = num
        self.total  = total
        self.title  = title
        self._start = 0.0

    def __enter__(self) -> "Step":
        self._start = time.time()
        print(f"\n  STEP {self.num}/{self.total} — {self.title.upper()}")
        return self

    def __exit__(self, exc_type, *_) -> None:
        elapsed = time.time() - self._start
        icon    = OK if exc_type is None else FAIL
        print(f"  {icon} Step {self.num} complete  ({_fmt_time(elapsed)})")
