import gc
import inspect
import sys
import functools
from typing import Callable
from pathlib import Path
from rich.console import Console
from rich.syntax import Syntax
from rich.text import Text
import textwrap

from memorylane.memory_readers.torch import get_memory_usage

# Initialize a shared Rich console and highlighter.
console = Console()


def profile(
    _fn: Callable | None = None, *, threshold: float = 0.5 * 1024**2
) -> Callable:  # noqa: D401
    """Decorator that prints memory usage after each executed *source line*.

    The wrapped function executes normally. Internally, a ``sys.settrace`` hook
    intercepts every *line* event originating from the function's source file.
    After the execution of each line, memory statistics are printed
    including the delta since the previous line, the delta in peak usage, and
    the current total allocated memory.

    Example
    -------
    >>> @profile
    ... def foo():
    ...     t = torch.randn(1024, 1024, device="cuda")
    ...     s = t.sum()
    ...     return s.item()
    """

    syntax = Syntax(
        "",
        "python",
        theme="monokai",
        line_numbers=False,
        word_wrap=False,
        indent_guides=True,
    )

    
    def _color_for_delta(d: float) -> str:
        if abs(d) < threshold:
            return "grey50"
        return "green bold" if d > 0 else "red bold"

    def decorator(fn: Callable) -> Callable:  # noqa: D401

        @functools.wraps(fn)
        def wrapper(*args, **kwargs):  # noqa: D401
            # Clear any residual allocations to start with a clean slate.
            gc.collect()

            filename = inspect.getsourcefile(fn)
            if filename is None:
                raise RuntimeError(
                    "Unable to determine source file for the traced function."
                )
            filename = Path(filename).resolve()

            source_lines, start_line = inspect.getsourcelines(fn)
            source_lines = textwrap.dedent("".join(source_lines)).splitlines()
            # Map absolute line numbers in the file -> source text.
            source_map: dict[int, str] = {
                start_line + idx: line.rstrip("\n")
                for idx, line in enumerate(source_lines)
            }
            func_display_name = getattr(fn, "__name__", str(fn))
            console.print(
                f"[bold]Tracing {func_display_name!r}[/bold] (file: {filename})"
            )

            baseline_mem, baseline_peak = get_memory_usage()
            prev_mem, prev_peak = baseline_mem, baseline_peak
            prev_file: Path | None = None
            prev_lineno: int | None = None

            def tracer(frame, event, arg):
                nonlocal prev_mem, prev_peak, prev_lineno, prev_file

                if (
                    (prev_file == filename) and
                    (prev_lineno in source_map.keys())
                ):
                    mem, peak = get_memory_usage()
                    mem -= baseline_mem
                    peak -= baseline_peak
                    delta_mem = mem - prev_mem
                    delta_peak = peak - prev_peak
                    prev_mem, prev_peak = mem, peak

                    is_significant = abs(delta_mem) > threshold or abs(delta_peak) > threshold

                    if is_significant or True:

                        mem_color, peak_color = map(
                            _color_for_delta,
                            (delta_mem, delta_peak)
                        )

                        segments: list[Text] = [
                            Text(f"Mem: {make_str(mem)}", style=mem_color),
                            Text(f"ΔMem: {make_str(delta_mem)}", style=mem_color),
                            Text(f"Peak: {make_str(peak)}", style=peak_color),
                            Text(f"ΔPeak: {make_str(delta_peak)}", style=peak_color),
                            Text(f"L{frame.f_lineno:<4}", style="cyan"),
                            syntax.highlight(source_map[prev_lineno]),
                        ]

                        console.print(*segments, sep=" | ", end="")

                prev_file = Path(frame.f_code.co_filename).resolve()
                prev_lineno = frame.f_lineno

                return tracer

            sys.settrace(tracer)
            try:
                return fn(*args, **kwargs)
            finally:
                sys.settrace(None)
                console.print(f"[bold]Done tracing {func_display_name!r}[/bold]")

        return wrapper

    # Support both @profile and @profile(...)
    if _fn is None:
        return decorator
    return decorator(_fn)


def make_str(mem: float) -> str:
    """Given a memory usage, in bytes, return a string suitable for printing."""
    return f"{mem / 1024**2:>6,.0f} MB"


if __name__ == "__main__":

    @profile
    def example_function():
        import torch

        a = torch.randn(8192, 8192, device="cuda")
        b = a @ a
        c = b.relu()
        if True:
            c *= c + 2 + 3
        # del a
        del b
        d = child_function(c)
        for i in range(3):
            c *= c
        torch.cuda.empty_cache()
        d = c.mean()
        out = d.item()

        return out

    @profile
    def child_function(x):
        y = x * x
        return y * (y + 2 + 3)

    example_function()
