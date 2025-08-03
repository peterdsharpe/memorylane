from asyncio import base_events
import gc
import inspect
import sys
import functools
from functools import partial
from typing import Callable, Literal
import threading
from pathlib import Path
from rich.console import Console  # type: ignore
from rich.syntax import Syntax  # type: ignore
from rich.text import Text  # type: ignore
import textwrap

# Initialize a shared Rich console and highlighter.
console = Console(
    force_jupyter=False,
    width=1000
)  # type: ignore

# Thread-local storage for current trace depth (used for indenting nested traces).
thread_data = threading.local()
thread_data._memorylane_indent_level = 0


def profile(
    _fn: Callable | None = None,
    *,
    memory_type: Literal["torch_cuda", "torch_cpu", "python"] = "torch_cuda",
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

    def decorator(fn: Callable) -> Callable:

        @functools.wraps(fn)
        def wrapper(*args, **kwargs):

            
            if memory_type == "torch_cuda":
                from memorylane.memory_readers.torch import get_memory_usage
                get_memory_usage = partial(get_memory_usage, device="cuda")  # type: ignore
                import torch
                torch.cuda.empty_cache()
            elif memory_type == "torch_cpu":
                from memorylane.memory_readers.torch import get_memory_usage
                get_memory_usage = partial(get_memory_usage, device="cpu")  # type: ignore
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
            indent_prefix: str = " " * 4 * thread_data._memorylane_indent_level
            
            def iprint(*args, **kwargs):
                """Print with indentation."""
                console.print(indent_prefix, *args, **kwargs)

            if thread_data._memorylane_indent_level == 0:
                iprint(
                    "[bold magenta]━━━━━━ MemoryLane: Line-by-Line Memory Profiler ━━━━━━[/bold magenta]"
                )

            thread_data._memorylane_indent_level += 1

            raw_filename = inspect.getsourcefile(fn)
            if raw_filename is None:
                raise RuntimeError(
                    "Unable to determine source file for the traced function."
                )
            filename: Path = Path(raw_filename).resolve()

            source_lines, start_line = inspect.getsourcelines(fn)

            dedented_source = textwrap.dedent("".join(source_lines)).splitlines()
            # Map absolute line numbers in the file -> source text.
            source_map: dict[int, str] = {
                start_line + idx: line for idx, line in enumerate(dedented_source)
            }
            func_raw_name = getattr(fn, "__name__", str(fn))
            func_display_name = f"[cyan]{func_raw_name!r}[/cyan]"
            file_display_name = f"[pale_turquoise4]{filename}:{start_line}[/pale_turquoise4]"
            iprint(
                f"[bold]Tracing {func_display_name}[/bold] (file: {file_display_name}):"
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

            # Pre-compute values for fast comparisons inside the tracer functions.
            target_code = fn.__code__  # type: ignore[attr-defined]
            target_filename = str(filename)

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
                if frame.f_code.co_filename != target_filename:
                    return None

                # Focus on *line* events (triggered before executing a source
                # line) and on the *return* event which fires immediately after
                # the return expression has been evaluated. All other events are
                # ignored, but we return the tracer so that execution within
                # this file remains traced.
                if event not in {"line", "return"}:
                    return _local_tracer

                # At this point we know we are inside our target file _and_ are
                # processing a line event.
                if (state["lineno"] is not None) and (state["lineno"] in source_map):
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
                        mem_color, peak_color = map(_color_for_delta, (delta_mem, delta_peak))

                        segments: list[Text] = [
                            Text(f"Mem: {make_str(mem)}", style=mem_color),
                            Text(f"ΔMem: {make_str(delta_mem)}", style=mem_color),
                            Text(f"Peak: {make_str(peak)}", style=peak_color),
                            Text(f"ΔPeak: {make_str(delta_peak)}", style=peak_color),
                            Text(
                                f"{filename.name}:{state['lineno']:<4}",
                                style="pale_turquoise4",
                            ),
                            syntax_highlighter.highlight(
                                source_map.get(state["lineno"], "<unknown>")
                            )[:-1],  # Strip trailing newline added by Syntax
                        ]

                        iprint(Text(" | ").join(segments))

                # Record *this* line as the new reference point.
                state["file"] = filename  # Constant within this tracer.
                state["lineno"] = frame.f_lineno

                return _local_tracer

            def _global_tracer(frame, event, arg):
                """Lightweight global tracer that *activates* the heavy tracer
                only when we enter the decorated function's frame.

                This design keeps the interpreter overhead negligible for all
                other Python code (especially inside external libraries) and is
                the key to maintaining acceptable performance when profiling
                through mechanisms like PyTorch's activation checkpointing.
                """

                # Activate the local tracer *only* for the decorated function's
                # frame. Returning ``None`` here ensures that *all* other
                # frames—including those inside heavy libraries—execute
                # entirely without tracing.
                if event == "call" and frame.f_code is target_code:
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
                thread_data._memorylane_indent_level -= 1
                # if thread_data._memorylane_indent_level != 0:
                #     iprint("\n")

        return wrapper

    # Support both @profile and @profile(...)
    if _fn is None:
        return decorator
    return decorator(_fn)


def make_str(mem: float) -> str:
    """Given a memory usage, in bytes, return a string suitable for printing."""
    return f"{mem / 1024**2:>6,.0f} MB"


if __name__ == "__main__":
    import torch  # type: ignore

    @profile
    def example_function():

        N = int((100 * (1024 ** 2) // 4) ** 0.5)

        x = torch.randn(N, N, device="cuda")
        x = x @ x
        x = x.relu()
        x = child_function(x)
        out = x.mean()

        return out

    @profile
    def child_function(x):
        return x * x + 5 * x


    example_function()

    from torch.utils.checkpoint import checkpoint

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(10000, 10000)

        @profile
        def forward(self, x):
            a = torch.randn(1, 10000, device="cuda")
            a = self.linear(a)
            c = a.relu()
            d = c + torch.mean(x)
            return d

    model = Model().to("cuda")

    # model(torch.randn(10).to("cuda"))

    # # model(torch.randn(10).to("cuda"))
