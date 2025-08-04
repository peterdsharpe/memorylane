from memorylane import profile
import torch
from rich.console import Console
from pathlib import Path

console = Console(force_jupyter=False, width=142, record=True)  # type: ignore


@profile(_console=console)
def my_function():
    x = torch.randn(5120, 5120, device="cuda")
    x = x @ x
    x = x.relu()
    x = x.mean()
    return x

my_function()

folder = Path(__file__).parent

console.save_html(folder / "memorylane_report.html", clear=False)
console.save_text(folder / "memorylane_report.txt", clear=False)
console.save_svg(folder / "memorylane_report.svg", clear=False, title="MemoryLane Report")