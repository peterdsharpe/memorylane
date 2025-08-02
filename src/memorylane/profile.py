import gc
import inspect
import sys
import functools
from typing import Callable
import threading
from pathlib import Path
from rich.console import Console  # type: ignore
from rich.syntax import Syntax  # type: ignore
from rich.text import Text  # type: ignore
import textwrap

from memorylane.memory_readers.torch import get_memory_usage

# Initialize a shared Rich console and highlighter.
console = Console()  # type: ignore

# Thread-local storage for current trace depth (used for indenting nested traces).
thread_data = threading.local()


def profile(
    _fn: Callable | None = None,
    *,
    threshold: float = 0.5 * 1024**2,
    only_show_significant: bool = False,
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
            return "grey30"
        return "green bold" if d > 0 else "red bold"

    def decorator(fn: Callable) -> Callable:

        @functools.wraps(fn)
        def wrapper(*args, **kwargs):

            # Determine indentation level for this invocation.
            indent_level: int = getattr(thread_data, "level", 0)
            indent_prefix: str = " " * (4 * indent_level)

            def iprint(*args, **kwargs):
                console.print(indent_prefix, *args, **kwargs)

            # Increment depth for child calls.
            thread_data.level = indent_level + 1

            filename = inspect.getsourcefile(fn)
            if filename is None:
                raise RuntimeError(
                    "Unable to determine source file for the traced function."
                )
            filename = Path(filename).resolve()

            source_lines, start_line = inspect.getsourcelines(fn)

            # Skip the function signature line.
            source_lines = source_lines[1:]
            start_line += 1

            dedented_source = textwrap.dedent("".join(source_lines)).splitlines()
            # Map absolute line numbers in the file -> source text (no trailing newlines).
            source_map: dict[int, str] = {
                start_line + idx: line for idx, line in enumerate(dedented_source)
            }
            func_raw_name = getattr(fn, "__name__", str(fn))
            func_display_name = f"[cyan]{func_raw_name!r}[/cyan]"
            file_display_name = f"[pale_turquoise4]{filename}:{start_line}[/pale_turquoise4]"
            iprint(
                f"[bold]Tracing {func_display_name}[/bold] (file: {file_display_name})"
            )

            # Clear any residual allocations to start with a clean slate.
            gc.collect()
            baseline_mem, baseline_peak = get_memory_usage()

            state = {
                "file": None,
                "lineno": None,
                "mem": baseline_mem,
                "peak": baseline_peak,
            }

            def tracer(frame, event, arg):

                if (state["file"] == filename) and (
                    state["lineno"] in source_map.keys()
                ):
                    mem, peak = get_memory_usage()
                    delta_mem = mem - state["mem"]
                    delta_peak = peak - state["peak"]
                    state["mem"], state["peak"] = mem, peak

                    is_significant = (
                        abs(delta_mem) > threshold or abs(delta_peak) > threshold
                    )

                    if not only_show_significant or is_significant:

                        mem_color, peak_color = map(
                            _color_for_delta, (delta_mem, delta_peak)
                        )

                        segments: list[Text] = [
                            Text(f"Mem: {make_str(mem)}", style=mem_color),
                            Text(f"ΔMem: {make_str(delta_mem)}", style=mem_color),
                            Text(f"Peak: {make_str(peak)}", style=peak_color),
                            Text(f"ΔPeak: {make_str(delta_peak)}", style=peak_color),
                            Text(f"L{state['lineno']:<4}", style="cyan"),
                            syntax.highlight(
                                source_map.get(state["lineno"], "<unknown>")
                            )[
                                :-1
                            ],  # trims newline
                        ]

                        line = Text(" | ").join(segments)
                        iprint(line)

                state["file"] = Path(frame.f_code.co_filename).resolve()
                state["lineno"] = frame.f_lineno

                return tracer

            prev_tracer = sys.gettrace()
            sys.settrace(tracer)
            try:
                return fn(*args, **kwargs)
            finally:
                # Restore whichever tracer was previously registered (could be None).
                sys.settrace(prev_tracer)
                # iprint(f"[bold]Done tracing {func_display_name}[/bold]")

                # Decrement depth when exiting this function.
                thread_data.level = indent_level

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
        import torch  # type: ignore

        a = torch.randn(8192, 8192, device="cuda")
        b = a @ a
        c = b.relu()
        if True:
            c *= c + 2 + 3
        # del a
        del b
        d = child_function(c)
        e = grandchild_function(d)
        for i in range(3):
            c *= c
        torch.cuda.empty_cache()
        d = c.mean()
        out = d.item()

        return out

    @profile
    def child_function(x):
        y = x * x + grandchild_function(x)
        return y * (y + 2 + 3)

    @profile
    def grandchild_function(x):
        n = x + x
        return n / n

    example_function()
