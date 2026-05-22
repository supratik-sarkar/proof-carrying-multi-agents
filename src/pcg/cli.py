"""
PCG-MAS CLI.

Thin Typer wrapper around the reviewer-facing scripts. The CLI keeps the same
entrypoints as the Makefile:

    pcg version
    pcg preflight
    pcg preflight40
    pcg full
    pcg figures
    pcg tables
    pcg run-r1 --config configs/r1_hotpotqa.yaml --seeds 0 1 2
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

try:
    import typer
    _HAS_TYPER = True
except ImportError:
    _HAS_TYPER = False


ROOT = Path(__file__).resolve().parents[2]


def _run_python_script(path: Path, args: list[str] | None = None) -> int:
    cmd = [sys.executable, str(path)]
    if args:
        cmd.extend(args)
    return subprocess.call(cmd, cwd=ROOT)


if _HAS_TYPER:
    app = typer.Typer(
        no_args_is_help=True,
        add_completion=False,
        pretty_exceptions_enable=False,
    )

    @app.command()
    def version() -> None:
        """Print the package version."""
        import pcg
        print(f"pcg {pcg.__version__}")

    @app.command()
    def preflight(
        n: int = typer.Option(10, help="Examples per cell."),
        seeds: list[int] = typer.Option([0], help="Random seeds."),
    ) -> None:
        """Run the two-cell preflight gate."""
        path = ROOT / "scripts" / "runs" / "run_preflight.py"
        argv = ["--n", str(n), "--seeds", *map(str, seeds)]
        raise typer.Exit(_run_python_script(path, argv))

    @app.command(name="preflight40")
    def preflight40(
        n: int = typer.Option(5, help="Examples per cell."),
        seeds: list[int] = typer.Option([0], help="Random seeds."),
    ) -> None:
        """Run all forty cells with a tiny budget."""
        path = ROOT / "scripts" / "runs" / "run_preflight_40_cells.py"
        argv = ["--n", str(n), "--seeds", *map(str, seeds)]
        raise typer.Exit(_run_python_script(path, argv))

    @app.command()
    def full(
        n: int = typer.Option(500, help="Examples per cell."),
        seeds: list[int] = typer.Option([0, 1, 2, 3], help="Random seeds."),
        allow_full_run: bool = typer.Option(False, help="Required guard for the full run."),
    ) -> None:
        """Run the full local forty-cell benchmark."""
        path = ROOT / "scripts" / "runs" / "run_local_40_cells.py"
        argv = ["--n", str(n), "--seeds", *map(str, seeds)]
        if allow_full_run:
            argv.append("--allow-full-run")
        raise typer.Exit(_run_python_script(path, argv))

    @app.command(name="run-r1")
    def run_r1(
        config: str = typer.Option(..., help="Path to YAML config."),
        seeds: list[int] = typer.Option([0, 1, 2, 3], help="Random seeds."),
        n_examples: int | None = typer.Option(None, help="Cap on examples per seed."),
        backend: str = typer.Option("mock", help="mock | hf_local | hf_inference"),
    ) -> None:
        """Run R1 audit decomposition / checkability."""
        from scripts.run_r1_checkability import main
        main(config=config, seeds=seeds, n_examples=n_examples, backend=backend)

    @app.command(name="run-r2")
    def run_r2(
        config: str = typer.Option(..., help="Path to YAML config."),
        seeds: list[int] = typer.Option([0, 1, 2, 3]),
        k_values: list[int] = typer.Option([1, 2, 4, 8]),
        n_examples: int | None = typer.Option(None),
        backend: str = typer.Option("mock"),
    ) -> None:
        """Run R2 redundancy law."""
        from scripts.run_r2_redundancy import main
        main(
            config=config,
            seeds=seeds,
            k_values=k_values,
            n_examples=n_examples,
            backend=backend,
        )

    @app.command(name="run-r3")
    def run_r3(
        config: str = typer.Option(..., help="Path to YAML config."),
        seeds: list[int] = typer.Option([0, 1, 2, 3]),
        n_examples: int | None = typer.Option(None),
        backend: str = typer.Option("mock"),
    ) -> None:
        """Run R3 replay-intervention responsibility."""
        from scripts.run_r3_responsibility import main
        main(config=config, seeds=seeds, n_examples=n_examples, backend=backend)

    @app.command(name="run-r4")
    def run_r4(
        config: str = typer.Option(..., help="Path to YAML config."),
        seeds: list[int] = typer.Option([0, 1, 2, 3]),
        eps_values: list[str] = typer.Option(["inf", "8", "3", "1"]),
        n_examples: int | None = typer.Option(None),
        backend: str = typer.Option("mock"),
    ) -> None:
        """Run R4 risk-control and privacy frontier."""
        from scripts.run_r4_risk_privacy import main
        main(
            config=config,
            seeds=seeds,
            eps_values=eps_values,
            n_examples=n_examples,
            backend=backend,
        )

    @app.command(name="run-r5")
    def run_r5(
        config: str = typer.Option(..., help="Path to YAML config."),
        seeds: list[int] = typer.Option([0, 1, 2, 3]),
        n_examples: int | None = typer.Option(None),
        backend: str = typer.Option("mock"),
    ) -> None:
        """Run R5 overhead accounting."""
        from scripts.run_r5_overhead import main
        main(config=config, seeds=seeds, n_examples=n_examples, backend=backend)

    @app.command(name="figures")
    def figures() -> None:
        """Regenerate all figures under results/figures."""
        path = ROOT / "scripts" / "figures" / "build_all_figures.py"
        raise typer.Exit(_run_python_script(path))

    @app.command(name="tables")
    def tables() -> None:
        """Regenerate CSV and LaTeX tables under results/tables."""
        path = ROOT / "scripts" / "tables" / "build_all_tables.py"
        raise typer.Exit(_run_python_script(path))

else:
    def app() -> None:
        if len(sys.argv) < 2:
            print("Usage: pcg <command> [options]")
            print("Install typer for the full CLI: pip install typer")
            sys.exit(1)
        print("Typer is not installed; use Makefile or scripts/* entrypoints.")
        sys.exit(1)


if __name__ == "__main__":
    app()
