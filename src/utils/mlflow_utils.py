"""Utilidades compartidas para configuración y registro de experimentos en MLflow."""

from __future__ import annotations

from pathlib import Path
import mlflow

DEFAULT_EXPERIMENT_NAME = "dengue_cr_spatiotemporal"


def init_mlflow(
    experiment_name: str = DEFAULT_EXPERIMENT_NAME,
    mlruns_dir: Path = Path("mlruns"),
) -> str:
    """Inicializa el tracking local de MLflow con backend SQLite dentro de mlruns/."""
    mlruns_dir = mlruns_dir.resolve()
    mlruns_dir.mkdir(parents=True, exist_ok=True)
    artifacts_dir = mlruns_dir / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    db_path = mlruns_dir / "mlflow.db"
    tracking_uri = f"sqlite:///{db_path.as_posix()}"
    mlflow.set_tracking_uri(tracking_uri)

    exp = mlflow.get_experiment_by_name(experiment_name)
    if exp is None:
        mlflow.create_experiment(
            name=experiment_name,
            artifact_location=artifacts_dir.as_uri(),
        )
    mlflow.set_experiment(experiment_name)
    return tracking_uri
