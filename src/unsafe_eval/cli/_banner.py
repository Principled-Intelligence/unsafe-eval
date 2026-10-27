"""The banner `unsafe-eval` prints when `run` starts and above its help."""

import sys
from importlib.metadata import version

_WIDE = """
██╗   ██╗███╗   ██╗███████╗ █████╗ ███████╗███████╗    ███████╗██╗   ██╗ █████╗ ██╗
██║   ██║████╗  ██║██╔════╝██╔══██╗██╔════╝██╔════╝    ██╔════╝██║   ██║██╔══██╗██║
██║   ██║██╔██╗ ██║███████╗███████║█████╗  █████╗█████╗█████╗  ██║   ██║███████║██║
██║   ██║██║╚██╗██║╚════██║██╔══██║██╔══╝  ██╔══╝╚════╝██╔══╝  ╚██╗ ██╔╝██╔══██║██║
╚██████╔╝██║ ╚████║███████║██║  ██║██║     ███████╗    ███████╗ ╚████╔╝ ██║  ██║███████╗
 ╚═════╝ ╚═╝  ╚═══╝╚══════╝╚═╝  ╚═╝╚═╝     ╚══════╝    ╚══════╝  ╚═══╝  ╚═╝  ╚═╝╚══════╝
""".strip("\n")

_NARROW = r"""
                          ____                           __
  __  ______  _________ _/ __/__        ___ _   ______ _/ /
 / / / / __ \/ ___/ __ `/ /_/ _ \______/ _ \ | / / __ `/ /
/ /_/ / / / (__  ) /_/ / __/  __/_____/  __/ |/ / /_/ / /
\__,_/_/ /_/____/\__,_/_/  \___/      \___/|___/\__,_/_/
""".strip("\n")


def print_banner() -> None:
    """Print the banner when stdout is a terminal, so redirected output stays plain."""
    if not sys.stdout.isatty():
        return
    from rich.console import Console
    from rich.text import Text

    console = Console()
    wide_enough = console.width >= max(len(line) for line in _WIDE.splitlines())
    console.print()
    # The report page's --accent color.
    console.print(Text(_WIDE if wide_enough else _NARROW, style="bold #7DF7D6"))
    console.print(" A framework for reproducible AI safety evaluation.")
    console.print(
        Text(f" unsafe-eval {version('unsafe-eval')} · Principled Intelligence", style="dim")
    )
    console.print()
