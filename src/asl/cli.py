"""`asl` - the single entry point to the pipeline.

asl data pull                     download the dataset from Kaggle
asl data sample --per-class 3     build the tiny demo/test subset
asl catalog --config ...          inspect the split before training anything
asl train --config configs/...    train whichever backend the config names
asl runs list / runs show         browse the immutable run directories
asl predict image / webcam        inference from a run
asl explain --run latest          Grad-CAM for a CNN run
asl serve --run latest            start the HTTP service
"""

from __future__ import annotations

import json
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from asl import __version__
from asl.config import ExperimentConfig
from asl.runs import list_runs, resolve_run

app = typer.Typer(add_completion=False, help="ASL fingerspelling recognition pipeline.")
data_app = typer.Typer(help="Dataset acquisition and sampling.")
runs_app = typer.Typer(help="Inspect training runs.")
predict_app = typer.Typer(help="Run inference from a trained run.")
app.add_typer(data_app, name="data")
app.add_typer(runs_app, name="runs")
app.add_typer(predict_app, name="predict")

console = Console()

ConfigOption = typer.Option("configs/landmark_rf.yaml", "--config", "-c", help="Experiment YAML.")
RunOption = typer.Option("latest", "--run", "-r", help="Run id, run path, or 'latest'.")
RunsDirOption = typer.Option("runs", "--runs-dir", help="Where run directories live.")


@app.callback(invoke_without_command=True)
def main(
    ctx: typer.Context,
    version: bool = typer.Option(False, "--version", help="Print the version and exit."),
) -> None:
    if version:
        console.print(f"asl-recognition-pipeline {__version__}")
        raise typer.Exit()
    if ctx.invoked_subcommand is None:
        console.print(ctx.get_help())


@data_app.command("pull")
def data_pull(
    destination: Path = typer.Option(Path("data/raw"), "--dest"),
    link: bool = typer.Option(False, "--link", help="Symlink the Kaggle cache instead of copying."),
) -> None:
    """Download the ASL Alphabet dataset from Kaggle."""
    from asl.data.download import pull_dataset

    root = pull_dataset(destination, link=link)
    console.print(f"[green]dataset ready[/green] -> {root}")
    console.print("Point `data.root` in your config at that path.")


@data_app.command("sample")
def data_sample(
    source: Path = typer.Option(Path("data/raw/asl_alphabet_train/asl_alphabet_train"), "--source"),
    destination: Path = typer.Option(Path("data/sample"), "--dest"),
    per_class: int = typer.Option(3, "--per-class"),
) -> None:
    """Build the small committed subset used by the tests and the demo."""
    from asl.data.download import build_sample

    out = build_sample(source, destination, per_class=per_class)
    console.print(f"[green]sample written[/green] -> {out}")


@app.command("catalog")
def catalog_command(config: Path = ConfigOption) -> None:
    """Scan the dataset and show the split, without training anything."""
    from asl.data.catalog import catalog_summary
    from asl.pipeline import build_catalog

    experiment = ExperimentConfig.from_yaml(config)
    catalog = build_catalog(experiment)
    summary = catalog_summary(catalog)

    table = Table(title=f"{experiment.name} - {summary['n_images']} images")
    table.add_column("class")
    table.add_column("train", justify="right")
    table.add_column("test", justify="right")
    counts = catalog.groupby(["label", "split"]).size().unstack(fill_value=0)
    for label, row in counts.iterrows():
        table.add_row(str(label), str(int(row.get("train", 0))), str(int(row.get("test", 0))))
    console.print(table)
    console.print(f"split strategy: [bold]{experiment.split.strategy}[/bold]")


@app.command("train")
def train_command(
    config: Path = ConfigOption,
    reuse_features: bool = typer.Option(
        True, "--reuse-features/--recompute-features", help="Reuse the cached landmark table."
    ),
) -> None:
    """Train the backend named by the config and write an immutable run."""
    from asl.pipeline import train

    experiment = ExperimentConfig.from_yaml(config)
    kwargs = {"reuse_features": reuse_features} if experiment.model == "landmark_rf" else {}
    run = train(experiment, log=console.print, **kwargs)
    console.print(f"[green]run complete[/green] -> {run.path}")


@runs_app.command("list")
def runs_list(runs_dir: Path = RunsDirOption) -> None:
    """List every run with its headline metric."""
    runs = list_runs(runs_dir)
    if not runs:
        console.print(f"no run under {runs_dir}")
        raise typer.Exit()

    table = Table(title=f"{len(runs)} runs")
    for column in ("run id", "backend", "split", "accuracy", "macro F1"):
        table.add_column(column)
    for run in runs:
        metrics = run.metrics()
        test = metrics.get("test", {})
        table.add_row(
            run.run_id,
            run.manifest().get("model", "?"),
            str(metrics.get("split_strategy", "?")),
            f"{test.get('accuracy', float('nan')):.4f}" if test else "-",
            f"{test.get('macro_f1', float('nan')):.4f}" if test else "-",
        )
    console.print(table)


@runs_app.command("show")
def runs_show(run: str = RunOption, runs_dir: Path = RunsDirOption) -> None:
    """Print the manifest and metrics of one run."""
    resolved = resolve_run(run, runs_dir)
    console.print_json(json.dumps(resolved.manifest(), indent=2))
    console.print_json(json.dumps(resolved.metrics(), indent=2, default=str))


@predict_app.command("image")
def predict_image(
    path: Path = typer.Argument(..., help="Image file."),
    run: str = RunOption,
    runs_dir: Path = RunsDirOption,
    top_k: int = typer.Option(3, "--top-k"),
) -> None:
    """Predict the letter shown in one image."""
    import cv2

    from asl.registry import load_recognizer_from_ref

    recognizer, resolved = load_recognizer_from_ref(run, runs_dir)
    image = cv2.imread(str(path))
    if image is None:
        console.print(f"[red]cannot read image:[/red] {path}")
        raise typer.Exit(code=1)

    prediction = recognizer.predict_image(image, top_k=top_k)
    if prediction is None:
        console.print("[yellow]no hand detected[/yellow]")
        raise typer.Exit(code=2)
    console.print(f"[bold]{prediction.label}[/bold]  ({prediction.confidence:.1%})")
    console.print(prediction.scores)
    console.print(f"run: {resolved.run_id}")


@predict_app.command("webcam")
def predict_webcam(
    run: str = RunOption,
    runs_dir: Path = RunsDirOption,
    source: int = typer.Option(0, "--source", help="Camera index."),
    hold: float = typer.Option(1.2, "--hold", help="Seconds a letter must be held to be typed."),
    min_confidence: float = typer.Option(0.6, "--min-confidence"),
) -> None:
    """Spell text from the webcam. 'q' quits, 'c' clears the buffer."""
    from asl.realtime import run_webcam
    from asl.registry import load_recognizer_from_ref

    recognizer, _ = load_recognizer_from_ref(run, runs_dir)
    text = run_webcam(recognizer, source=source, hold_seconds=hold, min_confidence=min_confidence)
    console.print(f"composed: [bold]{text}[/bold]")


@app.command("explain")
def explain_command(
    run: str = RunOption,
    runs_dir: Path = RunsDirOption,
    samples: int = typer.Option(9, "--samples", help="Number of test images to explain."),
) -> None:
    """Grad-CAM and feature maps for a CNN run (written into the run's artifacts)."""
    import pandas as pd
    import torch

    from asl.explain import GradCAM, feature_maps, overlay_heatmap, save_grid, save_image
    from asl.models.cnn import CatalogImageDataset, build_transforms
    from asl.registry import load_recognizer

    resolved = resolve_run(run, runs_dir)
    if resolved.manifest()["model"] != "cnn":
        console.print("[red]explain only applies to CNN runs[/red]")
        raise typer.Exit(code=1)

    recognizer = load_recognizer(resolved)
    catalog = pd.read_csv(resolved.artifacts / "catalog.csv")
    test = catalog[catalog["split"] == "test"].sample(n=min(samples, len(catalog)), random_state=0)
    dataset = CatalogImageDataset(
        test, recognizer.classes, build_transforms(recognizer.config, train=False)
    )

    out_dir = resolved.artifacts / "explain"
    with GradCAM(recognizer.model, recognizer.model.conv[-1]) as cam_builder:
        for i in range(len(dataset)):
            image, label = dataset[i]
            batch = image.unsqueeze(0).to(recognizer.device)
            cam = cam_builder.generate(batch)
            overlay = overlay_heatmap(image.numpy(), cam)
            save_image(
                overlay,
                out_dir / f"gradcam_{i:02d}_{recognizer.classes[label]}.png",
                title=f"Grad-CAM - true: {recognizer.classes[label]}",
            )

    image, _ = dataset[0]
    with torch.no_grad():
        for index, maps in enumerate(
            feature_maps(recognizer.model, image.unsqueeze(0).to(recognizer.device))
        ):
            save_grid(maps, out_dir / f"featuremaps_conv{index + 1}.png", title=f"conv {index + 1}")
    console.print(f"[green]explanations written[/green] -> {out_dir}")


@app.command("serve")
def serve_command(
    run: str = RunOption,
    runs_dir: Path = RunsDirOption,
    host: str = typer.Option("0.0.0.0", "--host"),  # noqa: S104 - container default
    port: int = typer.Option(8000, "--port"),
) -> None:
    """Start the inference API on a given run."""
    import os

    import uvicorn

    os.environ["ASL_RUN"] = run
    os.environ["ASL_RUNS_DIR"] = str(runs_dir)
    uvicorn.run("asl.api.main:app", host=host, port=port, reload=False)


if __name__ == "__main__":  # pragma: no cover
    app()
