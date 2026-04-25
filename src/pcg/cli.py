"""
PCG-MAS CLI.

Most experiment scripts live in `scripts/`; this module is a thin Typer
wrapper so the Makefile and external users have one consistent CLI.

Usage examples:
    pcg version
    pcg smoke
    pcg run-r1 --config configs/r1_hotpotqa.yaml --seeds 0 1 2
"""
from __future__ import annotations

import sys
from pathlib import Path

try:
    import typer
    _HAS_TYPER = True
except ImportError:
    _HAS_TYPER = False


if _HAS_TYPER:
    app = typer.Typer(no_args_is_help=True, add_completion=False, pretty_exceptions_enable=False)

    @app.command()
    def version() -> None:
        """Print the package version."""
        import pcg
        print(f"pcg {pcg.__version__}")

    @app.command()
    def smoke() -> None:
        """Run the Phase B+C smoke test (no model download)."""
        smoke_path = Path(__file__).parent.parent.parent / "scripts" / "test_phase_bc.py"
        sys.exit(_run_python_script(smoke_path))

    @app.command(name="run-r1")
    def run_r1(
        config: str = typer.Option(..., help="Path to YAML config"),
        seeds: list[int] = typer.Option([0, 1, 2, 3, 4], help="Random seeds"),
        n_examples: int | None = typer.Option(None, help="Cap on examples per seed"),
        backend: str = typer.Option("mock", help="mock | hf_local | hf_inference"),
    ) -> None:
        """Run R1 (audit-decomposition / checkability)."""
        from scripts.run_r1_checkability import main
        main(config=config, seeds=seeds, n_examples=n_examples, backend=backend)

    @app.command(name="run-r2")
    def run_r2(
        config: str = typer.Option(..., help="Path to YAML config"),
        seeds: list[int] = typer.Option([0, 1, 2, 3, 4]),
        k_values: list[int] = typer.Option([1, 2, 4, 8]),
        n_examples: int | None = typer.Option(None),
        backend: str = typer.Option("mock"),
    ) -> None:
        """Run R2 (redundancy law)."""
        from scripts.run_r2_redundancy import main
        main(config=config, seeds=seeds, k_values=k_values,
             n_examples=n_examples, backend=backend)

    @app.command(name="run-r3")
    def run_r3(
        config: str = typer.Option(...),
        seeds: list[int] = typer.Option([0, 1, 2, 3, 4]),
        n_examples: int | None = typer.Option(None),
        backend: str = typer.Option("mock"),
    ) -> None:
        """Run R3 (responsibility / diagnosis)."""
        from scripts.run_r3_responsibility import main
        main(config=config, seeds=seeds, n_examples=n_examples, backend=backend)

    @app.command(name="run-r4")
    def run_r4(
        config: str = typer.Option(...),
        seeds: list[int] = typer.Option([0, 1, 2, 3, 4]),
        eps_values: list[str] = typer.Option(["inf", "8", "3", "1"]),
        n_examples: int | None = typer.Option(None),
        backend: str = typer.Option("mock"),
    ) -> None:
        """Run R4 (risk + privacy)."""
        from scripts.run_r4_risk_privacy import main
        main(config=config, seeds=seeds, eps_values=eps_values,
             n_examples=n_examples, backend=backend)

    @app.command(name="run-r5")
    def run_r5(
        config: str = typer.Option(...),
        seeds: list[int] = typer.Option([0, 1, 2, 3, 4]),
        n_examples: int | None = typer.Option(None),
        backend: str = typer.Option("mock"),
    ) -> None:
        """Run R5 (overhead)."""
        from scripts.run_r5_overhead import main
        main(config=config, seeds=seeds, n_examples=n_examples, backend=backend)

    @app.command(name="make-figures")
    def make_figures(
        results_dir: str = typer.Option("results"),
        out: str = typer.Option("figures"),
    ) -> None:
        """Regenerate all figures from saved results."""
        from scripts.make_figures import main
        main(results_dir=results_dir, out=out)

    @app.command(name="make-tables")
    def make_tables(
        results_dir: str = typer.Option("results"),
        out: str = typer.Option("docs/tables"),
    ) -> None:
        """Regenerate LaTeX tables from saved results."""
        from scripts.make_tables import main
        main(results_dir=results_dir, out=out)

else:
    # Fallback when typer isn't installed: a tiny dispatcher
    def app() -> None:
        if len(sys.argv) < 2:
            print("Usage: pcg <command> [options]")
            print("(typer not installed — install with `pip install typer` for full CLI)")
            sys.exit(1)


def _run_python_script(path: Path) -> int:
    import subprocess
    return subprocess.call([sys.executable, str(path)])


if __name__ == "__main__":
    app()
