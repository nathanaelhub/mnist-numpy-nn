"""A dependency-free terminal progress bar for the training loop."""

import sys
import time
import shutil


class ProgressBar:
    """A Keras-style single-line progress bar.

    Call :meth:`update` after each mini-batch with the number of samples seen
    so far and a dict of running metrics; call :meth:`close` once at the end of
    the epoch to print the final newline.
    """

    def __init__(self, total, width=30, prefix=""):
        self.total = total
        self.width = width
        self.prefix = prefix
        self.start = time.time()
        self._last_len = 0

    def update(self, current, metrics=None):
        current = min(current, self.total)
        frac = current / self.total if self.total else 1.0
        filled = int(self.width * frac)
        bar = "=" * filled + (">" if filled < self.width else "") + \
              "." * (self.width - filled - (1 if filled < self.width else 0))

        elapsed = time.time() - self.start
        rate = current / elapsed if elapsed > 0 else 0.0
        eta = (self.total - current) / rate if rate > 0 else 0.0

        parts = [
            f"{self.prefix}",
            f"{current:>{len(str(self.total))}}/{self.total}",
            f"[{bar}]",
            f"{elapsed:5.1f}s",
        ]
        if current < self.total:
            parts.append(f"ETA {eta:4.1f}s")
        if metrics:
            parts.append("- " + " - ".join(f"{k}: {v:.4f}" for k, v in metrics.items()))

        line = " ".join(parts)
        # Pad so leftovers from a longer previous line are overwritten.
        pad = max(0, self._last_len - len(line))
        self._last_len = len(line)

        # Stay inside the terminal width when we can detect it.
        cols = shutil.get_terminal_size((120, 20)).columns
        sys.stdout.write("\r" + line[:cols] + " " * pad)
        sys.stdout.flush()

    def close(self):
        sys.stdout.write("\n")
        sys.stdout.flush()
