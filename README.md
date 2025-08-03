<h1 align="center"> MemoryLane 💾🛣️ </h1>

by Peter Sharpe

-----

A super-lightweight line-by-line memory profiler for numerical Python code. See where those pesky allocations are coming from!
* Supports [PyTorch](https://pytorch.org/) CUDA memory measurement, and more to come.
* Minimal dependencies (just [Rich](https://github.com/Textualize/rich) + your favorite numerical library)

## Installation

```bash
pip install memorylane
pip install memorylane[torch]  # For PyTorch support
```

## Usage

To use MemoryLane, just import it and decorate your function with `@profile`:

```python
import torch
from memorylane import profile

@profile
def my_function():
    x = torch.randn(5120, 5120, device="cuda")
    x = x @ x
    x = x.relu()
    x = x.mean()
    return x

my_function()
```

This will print your line-by-line memory usage:

<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8">
<style>
.r1 {color: #800080; text-decoration-color: #800080; font-weight: bold}
.r2 {font-weight: bold}
.r3 {color: #008080; text-decoration-color: #008080; font-weight: bold}
.r4 {color: #5f8787; text-decoration-color: #5f8787}
.r5 {color: #5f8787; text-decoration-color: #5f8787; font-weight: bold}
.r6 {color: #008000; text-decoration-color: #008000; font-weight: bold}
.r7 {color: #f8f8f2; text-decoration-color: #f8f8f2; background-color: #272822}
.r8 {color: #ff4689; text-decoration-color: #ff4689; background-color: #272822}
.r9 {color: #ae81ff; text-decoration-color: #ae81ff; background-color: #272822}
.r10 {color: #e6db74; text-decoration-color: #e6db74; background-color: #272822}
.r11 {color: #7f7f7f; text-decoration-color: #7f7f7f}
.r12 {color: #800000; text-decoration-color: #800000; font-weight: bold}
.r13 {color: #66d9ef; text-decoration-color: #66d9ef; background-color: #272822}
body {
    color: #000000;
    background-color: #ffffff;
}
/* Disable wrapping in <pre> and <code> blocks */
pre, code {
    white-space: pre;
    overflow-x: auto;
}
</style>
</head>
<body>
    <pre style="font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><code style="font-family:inherit"> <span class="r1">━━━━━━ MemoryLane: Line-by-Line Memory Profiler ━━━━━━</span>
 <span class="r2">Tracing </span><span class="r3">&#x27;my_function&#x27;</span> <span class="r2">(</span>file: <span class="r4">/home/psharpe/GitHub/memorylane/examples/make_report/make_reports.py:</span><span class="r5">9</span><span class="r2">)</span>:
 <span class="r6">Mem:    100 MB</span> | <span class="r6">ΔMem:    100 MB</span> | <span class="r6">Peak:    100 MB</span> | <span class="r6">ΔPeak:    100 MB</span> | <span class="r4">make_reports.py:11  </span> | <span class="r7">    x </span><span class="r8">=</span><span class="r7"> torch</span><span class="r8">.</span><span class="r7">randn(</span><span class="r9">5120</span><span class="r7">, </span><span class="r9">5120</span><span class="r7">, device</span><span class="r8">=</span><span class="r10">&quot;cuda&quot;</span><span class="r7">)</span>
 <span class="r6">Mem:    108 MB</span> | <span class="r6">ΔMem:      8 MB</span> | <span class="r6">Peak:    208 MB</span> | <span class="r6">ΔPeak:    108 MB</span> | <span class="r4">make_reports.py:12  </span> | <span class="r7">    x </span><span class="r8">=</span><span class="r7"> x </span><span class="r8">@</span><span class="r7"> x</span>
 <span class="r11">Mem:    108 MB</span> | <span class="r11">ΔMem:      0 MB</span> | <span class="r11">Peak:    208 MB</span> | <span class="r11">ΔPeak:      0 MB</span> | <span class="r4">make_reports.py:13  </span> | <span class="r7">    x </span><span class="r8">=</span><span class="r7"> x</span><span class="r8">.</span><span class="r7">relu()</span>
 <span class="r12">Mem:      8 MB</span> | <span class="r12">ΔMem:   -100 MB</span> | <span class="r11">Peak:    208 MB</span> | <span class="r11">ΔPeak:      0 MB</span> | <span class="r4">make_reports.py:14  </span> | <span class="r7">    x </span><span class="r8">=</span><span class="r7"> x</span><span class="r8">.</span><span class="r7">mean()</span>
 <span class="r11">Mem:      8 MB</span> | <span class="r11">ΔMem:      0 MB</span> | <span class="r11">Peak:    208 MB</span> | <span class="r11">ΔPeak:      0 MB</span> | <span class="r4">make_reports.py:15  </span> | <span class="r7">    </span><span class="r13">return</span><span class="r7"> x</span>
</code></pre>
</body>
</html>

## Features

* For complicated functions, filter the report to only show lines with non-negligible changes in memory usage: `@profile(only_show_significant=True)`
* When used from terminal, the printouts like `make_reports.py:11` become clickable links that will take you directly to the offending line in your code
* Profiling of multiple functions, including nested ones (these will be shown with indentation, to allow you to see where the allocations are coming from)
* Report generation in HTML and text formats
* (Work in progress) Support for measuring memory usage of:
    * PyTorch CPU operations
    * NumPy operations
    * JAX operations
    * Python whole-process memory usage
    * ...and more!

