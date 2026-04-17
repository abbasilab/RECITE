"""
recite CLI: root app with crawl and benchmark sub-commands.
"""
import typer
from dotenv import load_dotenv

from recite.utils.logging_config import configure_logging

load_dotenv()  # Load .env so UCSF_* and other vars are available for serve/crawl/benchmark

app = typer.Typer(name="recite", help="RECITE: Revising Eligibility Criteria Incorporating Textual Evidence.")


@app.callback()
def _ensure_logging(
    log_level: str = typer.Option(
        "INFO",
        "--log-level",
        help="Logging level: TRACE, DEBUG, INFO, WARNING, ERROR. Use DEBUG to see per-sample progress and retries.",
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="Shortcut for --log-level DEBUG (more comprehensive stdout logging).",
    ),
):
    """Configure logging to logs/ and stderr before any command."""
    level = "DEBUG" if verbose else log_level.upper()
    configure_logging(level=level, app_name="recite", also_stderr=True)


def _register_subapps():
    from recite.cli import benchmark as benchmark_mod
    app.add_typer(benchmark_mod.app, name="benchmark")


_register_subapps()
