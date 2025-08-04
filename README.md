<h1 align="center"> MemoryLane 💾🛣️ </h1>

[![PyPI](https://img.shields.io/pypi/v/memorylane.svg)](https://pypi.python.org/pypi/memorylane)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-brightgreen.svg)](https://opensource.org/license/apache-2-0)

by Peter Sharpe

-----

A super-lightweight line-by-line memory profiler for numerical Python code. See where those pesky allocations are coming from!
* Supports [PyTorch](https://pytorch.org/) CUDA memory measurement, and more to come.
* Minimal dependencies (just [Rich](https://github.com/Textualize/rich) + your favorite numerical library)

## Installation

```bash
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

![terminal](./examples/make_report/memorylane_report.svg)

## Features

* For complicated functions, filter the report to only show lines with non-negligible changes in memory usage: `@profile(only_show_significant=True)`
* When used from terminal via most editors (e.g., VSCode/Cursor, PyCharm, etc.), the printouts like `make_reports.py:11` become clickable links that will take you directly to the offending line in your code
* Profiling of multiple functions, including nested ones (these will be shown with indentation, to allow you to see where the allocations are coming from)
* Report generation in HTML and text formats
* (Work in progress) Support for measuring memory usage of:
    * PyTorch CPU operations
    * NumPy operations
    * JAX operations
    * Python whole-process memory usage
    * ...and more!

## Examples

Under construction - for now, see the [examples](./examples) folder!