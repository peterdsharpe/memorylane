import gc
import inspect
import sys
import functools
from functools import partial
from typing import Callable, Literal
from pathlib import Path
from rich.console import Console
from rich.syntax import Syntax
from rich.text import Text
import textwrap
from functools import lru_cache
from contextvars import ContextVar

# Initialize a shared Rich console and highlighter.
default_console = Console(force_jupyter=False, width=1000, record=True)

_indent_level: ContextVar[int] = ContextVar(
    "memorylane_indent_level", default=0
)


def profile(
    _fn: Callable | None = None,
    *,
    memory_type: Literal["torch_cuda", "torch_cpu", "python"] = "torch_cuda",
    threshold: float = 0.5 * 1024**2,
    only_show_significant: bool = False,
    _console: Console | None = None,
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
    if _console is None:
        _console = default_console

    def decorator(fn: Callable) -> Callable:

        @functools.wraps(fn)
        def wrapper(*args, **kwargs):

            if memory_type == "torch_cuda":
                from memorylane.memory_readers.torch import get_memory_usage

                get_memory_usage = partial(get_memory_usage, device="cuda")  # ty: ignore[invalid-assignment]
                import torch

                torch.cuda.empty_cache()
            elif memory_type == "torch_cpu":
                from memorylane.memory_readers.torch import get_memory_usage

                get_memory_usage = partial(get_memory_usage, device="cpu")  # ty: ignore[invalid-assignment]
            elif memory_type == "python":
                from memorylane.memory_readers.python import get_memory_usage
            else:
                raise ValueError(f"Invalid {memory_type=!r}.")

            syntax_highlighter = Syntax(
                code="",  # we'll call .highlight() later to generate the text
                lexer="python",
                theme="monokai",
                indent_guides=True,
            )

            def _color_for_delta(delta_mem: float) -> str:
                if abs(delta_mem) < threshold:
                    return "dim"
                return "green bold" if delta_mem > 0 else "red bold"

            # Determine indentation level for this invocation.
            indent_level: int = _indent_level.get()
            indent_prefix: str = " " * 4 * indent_level

            def iprint(*args, **kwargs):
                """Print with indentation."""
                _console.print(indent_prefix, *args, **kwargs)  # ty: ignore[possibly-unbound-attribute]

            if indent_level == 0:
                iprint(
                    "[bold magenta]━━━━━━ MemoryLane: Line-by-Line Memory Profiler ━━━━━━[/bold magenta]"
                )

            token = _indent_level.set(indent_level + 1)

            raw_filename = inspect.getsourcefile(fn)
            if raw_filename is None:
                raise RuntimeError(
                    "Unable to determine source file for the traced function."
                )
            fn_filepath: Path = Path(raw_filename).resolve()

            source_lines, start_line = inspect.getsourcelines(fn)

            dedented_source = textwrap.dedent("".join(source_lines)).splitlines()
            # Map absolute line numbers in the file -> source text.
            source_map: dict[int, str] = {
                start_line + idx: line for idx, line in enumerate(dedented_source)
            }
            fn_raw_name = getattr(fn, "__name__", str(fn))
            fn_display_name = f"[cyan]{fn_raw_name!r}[/cyan]"
            iprint(
                f"[bold]Tracing {fn_display_name}[/bold] (file: [pale_turquoise4]{fn_filepath}:{start_line}[/pale_turquoise4]):"
            )

            # Clear any residual allocations to start with a clean slate.
            gc.collect()

            baseline_mem, baseline_peak = get_memory_usage()

            state = {
                "lineno": None,
                "mem": baseline_mem,
                "peak": baseline_peak,
            }

            # Pre-compute values for fast comparisons inside the tracer functions.
            fn_code = fn.__code__  # ty: ignore[unresolved-attribute]
            fn_filepath_str = str(fn_filepath)
            fn_filename_str = str(fn_filepath.name)

            def _local_tracer(frame, event, arg):
                """Per-line tracer focusing exclusively on *this* source file.

                A trace function is invoked for every *event* in *every* frame once
                ``sys.settrace`` is active. We want to avoid doing any work for
                frames that do not originate from the decorated function's source
                file, as the overwhelming majority of executed Python byte-code
                belongs to external libraries (e.g. PyTorch).

                The strategy is therefore:
                1. **Early-exit** for frames from other files by returning ``None`` –
                   this disables tracing for that entire call-stack branch.
                2. Restrict heavy work (memory queries, rich rendering) to *line*
                   events within the file of interest.
                3. Maintain minimal state to compute deltas relative to the
                   *previous* executed line that we cared about.
                """
                # Fast-path exit for frames outside the decorated function's
                # file. Use raw-string comparison to avoid the overhead of
                # ``Path(...).resolve()`` which is surprisingly expensive when
                # triggered millions of times (e.g. inside PyTorch checkpoint
                # internals).
                if not (
                    (frame.f_code.co_filename == fn_filepath_str)  # Handles terminal
                    or (
                        frame.f_code.co_filename == fn_filename_str
                    )  # Handles IPython/Jupyter
                ):
                    return None

                # At this point we know we are inside our target file _and_ are
                # processing a line event.
                if (event in {"line", "return"}) and (state["lineno"] in source_map):
                    # Measure current memory usage and compute deltas relative to
                    # the previous interesting line.
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
                            Text(
                                f"{fn_filename_str}:{state['lineno']:<4}",
                                style="pale_turquoise4",
                            ),
                            syntax_highlighter.highlight(
                                source_map.get(state["lineno"], "<unknown>")
                            )[
                                :-1
                            ],  # Strip trailing newline added by Syntax
                        ]

                        iprint(Text(" | ").join(segments))

                state["lineno"] = frame.f_lineno

                return _local_tracer

            def _global_tracer(frame, event, arg) -> Callable | None:
                """Lightweight global tracer that *activates* the heavy tracer
                only when we enter the decorated function's frame.

                This design keeps the interpreter overhead negligible for all
                other Python code (especially inside external libraries) and is
                the key to maintaining acceptable performance when profiling
                through mechanisms like PyTorch's activation checkpointing.
                """

                # Activate the local tracer *only* for the decorated function's
                # frame. Returning ``None`` here ensures that *all* other
                # frames, including those inside heavy libraries, execute
                # entirely without tracing.
                if event == "call" and frame.f_code is fn_code:
                    return _local_tracer
                return None

            prev_tracer = sys.gettrace()
            sys.settrace(_global_tracer)
            try:
                return fn(*args, **kwargs)
            finally:
                # Restore whichever tracer was previously registered (could be None).
                sys.settrace(prev_tracer)

                # Decrement depth when exiting this function.
                _indent_level.reset(token)

        return wrapper

    # Support both @profile and @profile(...)
    if _fn is None:
        return decorator
    return decorator(_fn)


@lru_cache(maxsize=64)
def make_str(mem: float) -> str:
    """Given a memory usage, in bytes, return a string suitable for printing."""
    return f"{mem / 1024**2:>6,.0f} MB"


if __name__ == "__main__":
    import torch

    @profile
    def my_function():
        x = torch.randn(5120, 5120, device="cuda")
        x = x @ x
        x = x.relu()
        x = x.mean()
        return x

    my_function()
