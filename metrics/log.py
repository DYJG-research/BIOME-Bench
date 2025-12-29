from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any

from rich.console import Console


_CONSOLE = Console(highlight=False)


def _ts() -> str:
    return datetime.now().strftime("%H:%M:%S")


def _fmt_kv(kwargs: dict[str, Any]) -> str:
    if not kwargs:
        return ""
    parts: list[str] = []
    for k, v in kwargs.items():
        if v is None:
            continue
        s = str(v)
        parts.append(f"[dim]{k}=[/dim]{s}")
    return "  " + "  ".join(parts) if parts else ""


@dataclass(frozen=True)
class Logger:
    name: str

    def info(self, msg: str, **kwargs: Any) -> None:
        _CONSOLE.print(f"[cyan]{self.name}[/cyan] [dim]{_ts()}[/dim] {msg}{_fmt_kv(kwargs)}")

    def success(self, msg: str, **kwargs: Any) -> None:
        _CONSOLE.print(f"[green]{self.name}[/green] [dim]{_ts()}[/dim] [bold green]OK[/bold green] {msg}{_fmt_kv(kwargs)}")

    def warn(self, msg: str, **kwargs: Any) -> None:
        _CONSOLE.print(f"[yellow]{self.name}[/yellow] [dim]{_ts()}[/dim] [bold yellow]WARN[/bold yellow] {msg}{_fmt_kv(kwargs)}")

    def error(self, msg: str, **kwargs: Any) -> None:
        _CONSOLE.print(f"[red]{self.name}[/red] [dim]{_ts()}[/dim] [bold red]ERR[/bold red] {msg}{_fmt_kv(kwargs)}")


def get_logger(name: str) -> Logger:
    return Logger(name=name)
