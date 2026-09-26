"""File and folder scanning utilities for model_spark CLI."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from rich.console import Console
from rich.prompt import Confirm, Prompt
from rich.table import Table

console = Console()


def _scan_csv_files(directory: Path) -> list[Path]:
    """Scan directory for CSV and TSV files.
    
    Args:
        directory: Path to directory to scan
        
    Returns:
        List of CSV/TSV file paths sorted by name
    """
    if not directory.exists():
        raise FileNotFoundError(f"Directory does not exist: {directory}")
    if not directory.is_dir():
        raise NotADirectoryError(f"Path is not a directory: {directory}")
    
    csv_files = list(directory.glob("*.csv")) + list(directory.glob("*.tsv"))
    return sorted(set(csv_files))  # Remove duplicates and sort


def _scan_model_directories(directory: Path) -> list[Path]:
    """Scan directory for model_spark model folders.
    
    A valid model folder contains 'model_spark_metadata.json'.
    
    Args:
        directory: Path to directory to scan
        
    Returns:
        List of model directory paths sorted by name
    """
    if not directory.exists():
        raise FileNotFoundError(f"Directory does not exist: {directory}")
    if not directory.is_dir():
        raise NotADirectoryError(f"Path is not a directory: {directory}")
    
    model_dirs = [
        d for d in directory.iterdir()
        if d.is_dir() and (d / "model_spark_metadata.json").exists()
    ]
    return sorted(model_dirs)


def _display_file_table(files: list[Path], title: str) -> None:
    """Display files in a formatted table.
    
    Args:
        files: List of file paths to display
        title: Table title
    """
    table = Table(title=title)
    table.add_column("Index", style="cyan", width=8)
    table.add_column("Name", style="green")
    table.add_column("Size", justify="right")
    
    for idx, file_path in enumerate(files, 1):
        size_kb = file_path.stat().st_size / 1024
        size_str = f"{size_kb:.1f} KB" if size_kb < 1024 else f"{size_kb/1024:.1f} MB"
        table.add_row(str(idx), file_path.name, size_str)
    
    console.print(table)


def select_csv_file(directory: Optional[str] = None) -> str:
    """Interactively select a CSV/TSV file with rescan capability.
    
    Args:
        directory: Directory to scan (defaults to current directory)
        
    Returns:
        Path to selected CSV file
    """
    scan_dir = Path(directory or ".").expanduser().resolve()
    
    while True:
        try:
            csv_files = _scan_csv_files(scan_dir)
        except (FileNotFoundError, NotADirectoryError) as error:
            console.print(f"[bold red]Error:[/bold red] {error}")
            scan_dir = Path(
                Prompt.ask("Enter a valid directory path")
            ).expanduser().resolve()
            continue
        
        if not csv_files:
            console.print(
                f"[bold yellow]No CSV or TSV files found in {scan_dir}[/bold yellow]"
            )
            if Confirm.ask("Try a different directory?", default=True):
                scan_dir = Path(
                    Prompt.ask("Enter directory path")
                ).expanduser().resolve()
                continue
            # Fall back to manual path entry
            return Prompt.ask("Enter CSV/TSV file path")
        
        _display_file_table(csv_files, f"Available CSV/TSV files in {scan_dir}")
        
        while True:
            choice = Prompt.ask(
                "Select a file by index or enter 'rescan'/'other'",
                default="1"
            ).strip().lower()
            
            if choice == "rescan":
                break  # Rescan the directory
            elif choice == "other":
                # Allow manual path entry
                return Prompt.ask("Enter CSV/TSV file path")
            
            try:
                idx = int(choice) - 1
                if 0 <= idx < len(csv_files):
                    return str(csv_files[idx])
                console.print(
                    f"[bold red]Invalid index.[/bold red] Please enter a number between 1 and {len(csv_files)}."
                )
            except ValueError:
                console.print("[bold red]Invalid input.[/bold red] Enter a number, 'rescan', or 'other'.")


def select_model_directory(directory: Optional[str] = None) -> str:
    """Interactively select a model directory with rescan capability.
    
    Args:
        directory: Directory to scan (defaults to current directory)
        
    Returns:
        Path to selected model directory
    """
    scan_dir = Path(directory or ".").expanduser().resolve()
    
    while True:
        try:
            model_dirs = _scan_model_directories(scan_dir)
        except (FileNotFoundError, NotADirectoryError) as error:
            console.print(f"[bold red]Error:[/bold red] {error}")
            scan_dir = Path(
                Prompt.ask("Enter a valid directory path")
            ).expanduser().resolve()
            continue
        
        if not model_dirs:
            console.print(
                f"[bold yellow]No model_spark models found in {scan_dir}[/bold yellow]"
            )
            if Confirm.ask("Try a different directory?", default=True):
                scan_dir = Path(
                    Prompt.ask("Enter directory path")
                ).expanduser().resolve()
                continue
            # Fall back to manual path entry
            return Prompt.ask("Enter path to stored AutoGluon model directory")
        
        _display_file_table(model_dirs, f"Available models in {scan_dir}")
        
        while True:
            choice = Prompt.ask(
                "Select a model by index or enter 'rescan'/'other'",
                default="1"
            ).strip().lower()
            
            if choice == "rescan":
                break  # Rescan the directory
            elif choice == "other":
                # Allow manual path entry
                return Prompt.ask("Enter path to stored AutoGluon model directory")
            
            try:
                idx = int(choice) - 1
                if 0 <= idx < len(model_dirs):
                    return str(model_dirs[idx])
                console.print(
                    f"[bold red]Invalid index.[/bold red] Please enter a number between 1 and {len(model_dirs)}."
                )
            except ValueError:
                console.print("[bold red]Invalid input.[/bold red] Enter a number, 'rescan', or 'other'.")
