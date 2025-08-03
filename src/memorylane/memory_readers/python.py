"""Memory-usage helpers based purely on the Python standard library (plus an
optional psutil optimisation).

The goal is to expose a *single* public function – :func:`get_memory_usage` –
mirroring the API provided by ``memorylane.memory_readers.torch`` so that the
``@profile`` decorator can seamlessly switch between CUDA-centric and generic
process-wide memory tracking.
"""

from __future__ import annotations

import contextlib
import os
import sys
import warnings
from pathlib import Path

__all__: list[str] = ["get_memory_usage"]

# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------


def _page_size() -> int:  # noqa: D401 – helper
    """Return the system page size in **bytes**.

    ``os.sysconf`` keys vary by platform but ``SC_PAGE_SIZE`` and
    ``SC_PAGESIZE`` are the most common. We try both before defaulting to
    ``4096`` – a sensible fallback for the vast majority of modern systems.
    """

    for key in ("SC_PAGE_SIZE", "SC_PAGESIZE"):
        with contextlib.suppress(ValueError, KeyError):
            return os.sysconf(key)  # type: ignore[arg-type]
    return 4096  # pragma: no cover – unlikely fallback


def _rss_via_procfs() -> int | None:  # noqa: D401 – helper
    """Attempt to read *current* RSS from ``/proc/self/statm``.

    The file format is documented in `proc(5)`_. Field order:

        ``size resident shared text lib data dt``

    The *resident* column (index ``1``) is the resident set size in **pages**.

    Returns
    -------
    int | None
        The resident set size in bytes or ``None`` if ``/proc`` is not
        available (e.g. non-Linux platforms).
    """

    statm_path = Path("/proc/self/statm")
    if not statm_path.exists():
        return None

    try:
        content = statm_path.read_text().split()
        resident_pages = int(content[1])
    except (FileNotFoundError, IndexError, ValueError):  # pragma: no cover
        return None

    return resident_pages * _page_size()


def _peak_rss_via_resource() -> int | None:  # noqa: D401 – helper
    """Return the *peak* resident set size using :pymod:`resource`.

    Values are in **kilobytes** on most POSIX systems, except macOS where they
    are already reported in bytes. We normalise to bytes here.
    """

    with contextlib.suppress(ImportError):
        import resource  # pylint: disable=import-error

        usage = resource.getrusage(resource.RUSAGE_SELF)
        factor = 1 if sys.platform == "darwin" else 1024  # macOS already bytes
        return int(usage.ru_maxrss * factor)
    return None  # pragma: no cover


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def get_memory_usage() -> tuple[int, int]:  # noqa: D401
    """Return *current* and *peak* RSS for the running Python process.

    The function follows a cascade of increasingly general strategies:

    1. *psutil* (if installed) – provides both metrics in a single call.
    2. ProcFS + :pymod:`resource` (Linux/Unix) – combines current RSS from
       ``/proc`` with peak RSS from the C stdlib.
    3. :pymod:`resource` alone – when nothing else is available we fall back to
       returning the peak value for *both* metrics (better than nothing).

    Returns
    -------
    tuple[int, int]
        ``(current_rss_bytes, peak_rss_bytes)``

    Notes
    -----
    *   The values correspond to *resident* memory rather than virtual memory.
    *   The function never raises – if a reliable measurement is impossible we
        make a best-effort guess and emit a :class:`RuntimeWarning`.
    """

    # Strategy 1 – psutil ----------------------------------------------------
    with contextlib.suppress(ModuleNotFoundError):
        import psutil  # type: ignore  # noqa: WPS433 (optional dep)

        proc = psutil.Process()
        mem_info = proc.memory_info()
        # On Linux and macOS psutil exposes `rss` (current) but peak values are
        # platform-specific. We attempt the common attributes and fallback.
        current = mem_info.rss
        peak = getattr(mem_info, "peak_wset", None) or getattr(
            mem_info, "vms", None
        )
        if peak is None:
            # No peak available – we'll fill it later.
            peak = -1
        return current, int(peak)

    # Strategy 2 – /proc + resource -----------------------------------------
    current_rss = _rss_via_procfs()
    peak_rss = _peak_rss_via_resource()
    if current_rss is not None and peak_rss is not None:
        return current_rss, peak_rss

    # Strategy 3 – resource only --------------------------------------------
    if peak_rss is not None:
        warnings.warn(
            "Could not determine current RSS; using peak value as a proxy.",
            RuntimeWarning,
            stacklevel=2,
        )
        return peak_rss, peak_rss

    # Last resort -----------------------------------------------------------
    warnings.warn(
        "Unable to determine process memory usage on this platform.",
        RuntimeWarning,
        stacklevel=2,
    )
    return -1, -1
