"""TensorFlow 2 sequence adapter using tf.keras, imported only when explicitly requested.

The NumPy data contract exposes shapes and timing. Keras owns numerical training.
"""
from dataclasses import asdict, dataclass, field
from pathlib import Path
import json
import os
import numpy as np
from ml_lib.problems.forecasting import SequenceDataset


@dataclass(frozen=True)
class SequenceConfig:
    architecture: str
    hidden_units: int
    dense_units: int
    epochs: int
    batch_size: int
    learning_rate: float
    patience: int
    seed: int = 42
    cpu_only: bool = True
    deterministic: bool = True

    def __post_init__(self):
        if self.architecture not in {"gru", "lstm", "bilstm"}:
            raise ValueError("architecture must be 'gru', 'lstm' or 'bilstm'.")
        sizes = (self.hidden_units, self.dense_units, self.epochs, self.batch_size)
        if any(type(value) is not int or value < 1 for value in sizes):
            raise ValueError("Sequence dimensions and training budgets must be positive integers.")
        if (type(self.patience) is not int or self.patience < 0
                or not np.isfinite(self.learning_rate) or self.learning_rate <= 0):
            raise ValueError("Use nonnegative integer patience and a positive finite learning rate.")


@dataclass(frozen=True)
class TransformerConfig:
    """Choose attention capacity and training effort explicitly."""
    model_dim: int
    num_heads: int
    num_blocks: int
    feedforward_dim: int
    dropout: float
    dense_units: int
    epochs: int
    batch_size: int
    learning_rate: float
    patience: int
    seed: int = 42
    cpu_only: bool = True
    deterministic: bool = True
    architecture: str = field(default="transformer", init=False)

    def __post_init__(self):
        sizes = (self.model_dim, self.num_heads, self.num_blocks,
                 self.feedforward_dim, self.dense_units, self.epochs, self.batch_size)
        if any(type(value) is not int or value < 1 for value in sizes):
            raise ValueError("Transformer dimensions and training budgets must be positive integers.")
        if self.model_dim % self.num_heads or self.model_dim % 2:
            raise ValueError("model_dim must be even and divisible by num_heads.")
        if not np.isfinite(self.dropout) or not 0 <= self.dropout < 1:
            raise ValueError("dropout must be finite and in [0, 1).")
        if (type(self.patience) is not int or self.patience < 0
                or not np.isfinite(self.learning_rate) or self.learning_rate <= 0):
            raise ValueError("Use nonnegative integer patience and a positive finite learning rate.")


class SequenceForecaster:
    def __init__(self, config: SequenceConfig | TransformerConfig):
        self.config = config
        self.model = None
        self.history = {}
        self.scaling = {}
        self.signature = None

    def _tensorflow(self):
        # Must happen before importing TF. Restart the kernel after changing CPU policy.
        if self.config.cpu_only:
            os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
        if self.config.deterministic:
            os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")
            os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
        try:
            import tensorflow as tf
        except ImportError as exc:
            raise ImportError(
                "TensorFlow is not installed. In your activated .venv, run "
                "python -m pip install -r requirements.txt, then restart the kernel. "
                "No package installation was attempted automatically."
            ) from exc
        if self.config.cpu_only:
            try:
                tf.config.set_visible_devices([], "GPU")
            except RuntimeError as exc:
                raise RuntimeError("TensorFlow devices are already initialized. Restart the kernel to enforce CPU mode.") from exc
        tf.keras.utils.set_random_seed(self.config.seed)
        if self.config.deterministic:
            tf.config.experimental.enable_op_determinism()
        if isinstance(self.config, TransformerConfig):
            # Register our positional layer before building or safely loading it.
            from ml_lib.models import transformer_layers
        return tf

    @staticmethod
    def _signature(data: SequenceDataset) -> dict:
        return {"lookback": data.lookback,
                "history_names": list(data.history_names), "future_names": list(data.future_names)}

    def _check(self, data: SequenceDataset):
        if len(data.y) == 0:
            raise ValueError("No complete sequences. Inspect window coverage and split boundaries.")
        if self.signature is not None and self.signature != self._signature(data):
            raise ValueError("Sequence shape/feature contract differs from training.")

    def _fit_scaling(self, data: SequenceDataset):
        # Fit only on training windows. Repeated observations are weighted by their
        # occurrence in training windows. Nothing from validation/test is fitted.
        for key, array in (("history", data.history), ("future", data.future)):
            if array.shape[-1] == 0:
                self.scaling[key + "_mean"] = np.zeros((0,), np.float32)
                self.scaling[key + "_scale"] = np.ones((0,), np.float32)
            else:
                flat = array.reshape(-1, array.shape[-1]).astype(np.float64)
                self.scaling[key + "_mean"] = flat.mean(axis=0).astype(np.float32)
                std = flat.std(axis=0)
                self.scaling[key + "_scale"] = np.where(std < 1e-8, 1.0, std).astype(np.float32)
        self.scaling["y_mean"] = np.asarray(data.y.mean(dtype=np.float64), dtype=np.float32)
        self.scaling["y_scale"] = np.asarray(max(float(data.y.std(dtype=np.float64)), 1e-8), dtype=np.float32)

    def _inputs(self, history: np.ndarray, future: np.ndarray) -> dict:
        return {key: ((array - self.scaling[key + "_mean"]) / self.scaling[key + "_scale"]).astype(np.float32)
                for key, array in (("history", history), ("future", future))
                if array.shape[-1] > 0}

    def fit(self, data: SequenceDataset, validation: SequenceDataset) -> "SequenceForecaster":
        self.signature = self._signature(data)
        self._check(data); self._check(validation)
        if data.target_times.max() >= validation.target_times.min():
            raise ValueError("Training and validation target intervals must be disjoint and chronological.")
        tf = self._tensorflow()
        tf.keras.backend.clear_session()
        tf.keras.utils.set_random_seed(self.config.seed)
        self._fit_scaling(data)
        inputs = {}
        inputs["history"] = tf.keras.Input(shape=data.history.shape[1:], name="history")
        if isinstance(self.config, TransformerConfig):
            from ml_lib.models.transformer_layers import encode_history
            representation = encode_history(inputs["history"], self.config)
        else:
            recurrent = tf.keras.layers.GRU if self.config.architecture == "gru" else tf.keras.layers.LSTM
            layer = recurrent(self.config.hidden_units, name="temporal_representation")
            if self.config.architecture == "bilstm":
                layer = tf.keras.layers.Bidirectional(layer, merge_mode="concat", name="bidirectional_history")
            representation = layer(inputs["history"])
        if data.future.shape[-1]:
            inputs["future"] = tf.keras.Input(shape=data.future.shape[1:], name="future")
            representation = tf.keras.layers.Concatenate()([representation, inputs["future"]])
        hidden = tf.keras.layers.Dense(self.config.dense_units, activation="relu")(representation)
        output = tf.keras.layers.Dense(1, name="scaled_forecast")(hidden)
        self.model = tf.keras.Model(inputs=inputs, outputs=output)
        self.model.compile(optimizer=tf.keras.optimizers.Adam(self.config.learning_rate), loss="mse")

        def batches(dataset: SequenceDataset, shuffle: bool):
            labels = ((dataset.y - self.scaling["y_mean"]) / self.scaling["y_scale"]).astype(np.float32)[:, None]
            ds = tf.data.Dataset.from_tensor_slices((self._inputs(dataset.history, dataset.future), labels))
            if shuffle:
                ds = ds.shuffle(len(labels), seed=self.config.seed, reshuffle_each_iteration=True)
            # Shuffling already-built TRAIN windows is not shuffling time before splitting.
            options = tf.data.Options()
            options.threading.private_threadpool_size = 1
            options.threading.max_intra_op_parallelism = 1
            return ds.batch(self.config.batch_size).with_options(options).prefetch(1)

        stop = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=self.config.patience, restore_best_weights=True)
        result = self.model.fit(
            batches(data, True), validation_data=batches(validation, False),
            epochs=self.config.epochs, callbacks=[stop], verbose=0,
        )
        self.history = {key: [float(x) for x in values] for key, values in result.history.items()}
        return self

    def predict(self, data: SequenceDataset) -> np.ndarray:
        """Predict a labeled evaluation dataset using the input-only path."""
        if self.model is None:
            raise RuntimeError("Fit or load the sequence forecaster first.")
        self._check(data)
        return self.predict_inputs(
            data.history, data.future,
            history_names=data.history_names, future_names=data.future_names,
        )

    def predict_inputs(self, history, future, *, history_names, future_names) -> np.ndarray:
        """Predict from a 3D history array and 2D known-next-hour inputs.

        Inputs must have the fitted feature order and window lengths. With no
        known-future features, pass an array shaped [sample, 0].
        Callers remain responsible for field availability at forecast issuance.
        """
        if self.model is None or self.signature is None:
            raise RuntimeError("Fit or load the sequence forecaster first.")
        history = np.asarray(history, dtype=np.float32)
        future = np.asarray(future, dtype=np.float32)
        if history.ndim != 3 or future.ndim != 2:
            raise ValueError("Expected 3D history and 2D known-next-hour inputs.")
        n = history.shape[0]
        signature = self.signature
        if (
            n == 0
            or history.shape[1:] != (signature["lookback"], len(signature["history_names"]))
            or future.shape != (n, len(signature["future_names"]))
            or list(history_names) != signature["history_names"]
            or list(future_names) != signature["future_names"]
        ):
            raise ValueError("Input shape/feature contract differs from training, or inputs are empty.")
        if not np.isfinite(history).all() or not np.isfinite(future).all():
            raise ValueError("Forecast inputs must be finite.")
        inputs = self._inputs(history, future)
        outputs = []
        for start in range(0, n, self.config.batch_size):
            batch = {key: array[start:start + self.config.batch_size] for key, array in inputs.items()}
            outputs.append(np.asarray(self.model(batch, training=False)))
        scaled = np.concatenate(outputs, axis=0)[:, 0]
        return (scaled * self.scaling["y_scale"] + self.scaling["y_mean"]).astype(float)

    def describe(self) -> dict:
        description = {"backend": "TensorFlow 2 via tf.keras", "config": asdict(self.config), "signature": self.signature,
                "policies": {"optimizer": "Adam", "loss": "mse", "monitor": "val_loss",
                             "restore_best_weights": True, "shuffle_training_windows": True},
                "loss_space": "training-standardized target", "prediction_space": "original target units",
                "parameters": int(self.model.count_params()) if self.model is not None else None}
        if isinstance(self.config, TransformerConfig):
            description["representation"] = {
                "positions": "fixed sinusoidal", "normalization": "pre-layer",
                "pooling": "global average", "attention": "completed history only",
            }
        else:
            description["hidden_units_scope"] = "per direction" if self.config.architecture == "bilstm" else "single direction"
        return description

    def summary(self) -> str:
        if self.model is None:
            raise RuntimeError("Fit or load the model before requesting a summary.")
        lines = []
        self.model.summary(print_fn=lambda line, **kwargs: lines.append(line))
        return "\n".join(lines)

    def save(self, directory: str | Path) -> None:
        if self.model is None:
            raise RuntimeError("Cannot save an unfitted model.")
        directory = Path(directory); directory.mkdir(parents=True, exist_ok=True)
        self.model.save(directory / "model.keras")
        np.savez(directory / "scaling.npz", **self.scaling)
        (directory / "metadata.json").write_text(json.dumps({
            "config": asdict(self.config), "signature": self.signature, "history": self.history,
        }, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, directory: str | Path) -> "SequenceForecaster":
        """Load your own artifacts with safe Keras loading and registered layers."""
        directory = Path(directory)
        info = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
        config = dict(info["config"])
        if config.get("architecture") == "transformer":
            config.pop("architecture")
            instance = cls(TransformerConfig(**config))
        else:
            instance = cls(SequenceConfig(**config))
        tf = instance._tensorflow()
        instance.model = tf.keras.models.load_model(directory / "model.keras", safe_mode=True)
        with np.load(directory / "scaling.npz", allow_pickle=False) as arrays:
            instance.scaling = {key: arrays[key] for key in arrays.files}
        instance.signature, instance.history = info["signature"], info["history"]
        return instance


def build_sequence_forecaster(config: SequenceConfig | TransformerConfig) -> SequenceForecaster:
    return SequenceForecaster(config)
