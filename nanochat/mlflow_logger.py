"""
MLflow integration for nanochat training.
Drop-in replacement for wandb logging — same .log() / .finish() interface.

Usage:
  python -m scripts.base_train --tracker mlflow
  python -m scripts.base_train --tracker mlflow --mlflow-uri http://localhost:5000

Start MLflow UI:
  mlflow ui --port 5000
"""

import os
import json
import time


class MLflowLogger:
    """MLflow experiment tracker with the same interface as wandb/DummyWandb."""

    def __init__(self, experiment="nanochat", run_name="dummy", config=None,
                 tracking_uri=None):
        try:
            import mlflow
            import mlflow.pytorch
        except ImportError:
            raise ImportError(
                "mlflow is required for --tracker mlflow. "
                "Install with: pip install mlflow"
            )

        self._mlflow = mlflow

        # Set tracking URI (local file store by default)
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)
        elif os.environ.get("MLFLOW_TRACKING_URI"):
            pass  # already set in env
        # else: default file store under ./mlruns

        # Create or get experiment
        mlflow.set_experiment(experiment)

        # Start run
        tags = {"framework": "nanochat", "task": "pretraining"}
        self._run = mlflow.start_run(run_name=run_name, tags=tags)

        # Log config as params
        if config:
            # Flatten nested dicts, convert non-primitives to strings
            flat = {}
            for k, v in config.items():
                if isinstance(v, (str, int, float, bool)):
                    flat[k] = v
                elif v is not None:
                    flat[k] = str(v)
            mlflow.log_params(flat)

        self._step = 0
        self._last_log_time = time.time()

    def log(self, data=None, **kwargs):
        """Log metrics. Accepts dict or keyword arguments."""
        import mlflow

        metrics = {}
        if data and isinstance(data, dict):
            metrics.update(data)
        metrics.update(kwargs)

        if not metrics:
            return

        # Extract step if provided
        step = metrics.pop("step", self._step)

        # Convert non-numeric values to strings and log as params
        numeric_metrics = {}
        for k, v in metrics.items():
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                numeric_metrics[k] = float(v)
            elif isinstance(v, str):
                # Log string values as tags (not metrics)
                mlflow.set_tag(k, v)
            # Skip other types silently

        if numeric_metrics:
            mlflow.log_metrics(numeric_metrics, step=int(step))
            self._step = int(step) + 1

    def finish(self):
        """End the MLflow run."""
        if self._run:
            self._mlflow.end_run()
            self._run = None

    def __del__(self):
        self.finish()
