"""Backend 2 - end-to-end convolutional classifier (PyTorch).

Three conv blocks (conv -> ReLU -> max-pool -> batch-norm) then a two-layer head.
Small on purpose: it trains in a few epochs on one GPU, and the interest of the
repo is the pipeline around it, not the architecture. Unlike the landmark
backend it needs no hand detector, so it also predicts on frames where MediaPipe
finds nothing - and, for the same reason, happily predicts on frames with no
hand at all.

Torch is imported lazily so the landmark backend, the API and the tests run
without it installed.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from asl.config import CnnConfig
from asl.models.base import Prediction, Recognizer, _Timer

if TYPE_CHECKING:  # pragma: no cover
    import torch
    from torch import nn

WEIGHTS_FILE = "cnn.pt"
META_FILE = "cnn.json"


def resolve_device(preference: str = "auto") -> str:
    import torch

    if preference == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return preference


def build_model(n_classes: int, image_size: int = 100) -> nn.Module:
    """Convolutional stack. Shapes are derived, never hard-coded."""
    import torch
    from torch import nn

    conv = nn.Sequential(
        nn.Conv2d(3, 32, kernel_size=5),
        nn.ReLU(),
        nn.MaxPool2d(2, 2),
        nn.BatchNorm2d(32),
        nn.Conv2d(32, 64, kernel_size=3),
        nn.ReLU(),
        nn.MaxPool2d(2, 2),
        nn.BatchNorm2d(64),
        nn.Conv2d(64, 96, kernel_size=5),
        nn.ReLU(),
        nn.MaxPool2d(2, 2),
        nn.BatchNorm2d(96),
    )
    # Probe in eval mode: batch-norm refuses a batch of one while training.
    conv.eval()
    with torch.no_grad():
        flat = conv(torch.zeros(1, 3, image_size, image_size)).flatten(1).shape[1]
    conv.train()
    if flat == 0:
        raise ValueError(f"image_size={image_size} is too small for this architecture")

    class Classifier(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.conv = conv
            self.head = nn.Sequential(
                nn.Linear(flat, 128),
                nn.ReLU(),
                nn.LayerNorm(128),
                nn.Linear(128, n_classes),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            x = self.conv(x)
            return self.head(x.flatten(1))

    return Classifier()


def build_transforms(config: CnnConfig, train: bool):
    from torchvision import transforms

    steps = [transforms.Resize((config.image_size, config.image_size))]
    if train and config.augment:
        steps += [
            transforms.ColorJitter(brightness=0.25, contrast=0.25),
            transforms.RandomResizedCrop(config.image_size, scale=(0.9, 1.0)),
        ]
    steps.append(transforms.ToTensor())
    return transforms.Compose(steps)


class CatalogImageDataset:
    """Dataset driven by the catalog DataFrame, not by directory scanning.

    This is what keeps the train and test sets honest: both datasets read the same
    catalog, and the split lives in a column, so no image can silently move sides.
    """

    def __init__(self, frame: pd.DataFrame, classes: list[str], transform: Any) -> None:
        self.paths = frame["path"].tolist()
        self.labels = [classes.index(label) for label in frame["label"]]
        self.classes = classes
        self.transform = transform

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int):
        from PIL import Image

        with Image.open(self.paths[index]) as image:
            tensor = self.transform(image.convert("RGB"))
        return tensor, self.labels[index]


def train_cnn(
    catalog: pd.DataFrame,
    classes: list[str],
    config: CnnConfig,
    on_log: Callable[[str], None] = print,
) -> tuple[nn.Module, list[dict[str, float]]]:
    """Train the classifier and return it with the per-epoch history."""
    import torch
    from torch import nn
    from torch.utils.data import DataLoader

    torch.manual_seed(config.seed)
    device = resolve_device(config.device)

    train_frame = catalog[catalog["split"] == "train"]
    if train_frame.empty:
        raise ValueError("no training rows in the catalog")

    dataset = CatalogImageDataset(train_frame, classes, build_transforms(config, train=True))
    loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        drop_last=False,
    )

    model = build_model(len(classes), config.image_size).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    criterion = nn.CrossEntropyLoss()

    history: list[dict[str, float]] = []
    for epoch in range(1, config.epochs + 1):
        model.train()
        running, seen, correct = 0.0, 0, 0
        for batch, (images, targets) in enumerate(loader):
            images, targets = images.to(device), torch.as_tensor(targets).to(device)
            outputs = model(images)
            loss = criterion(outputs, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running += float(loss.item()) * images.size(0)
            seen += images.size(0)
            correct += int((outputs.argmax(1) == targets).sum().item())
            if config.log_every and batch % config.log_every == 0:
                on_log(f"epoch {epoch} batch {batch} loss {loss.item():.4f}")

        history.append(
            {
                "epoch": epoch,
                "train_loss": round(running / max(seen, 1), 4),
                "train_accuracy": round(correct / max(seen, 1), 4),
            }
        )
        on_log(f"epoch {epoch} done: {history[-1]}")

    return model, history


def predict_frame_batch(
    model: nn.Module,
    frame: pd.DataFrame,
    classes: list[str],
    config: CnnConfig,
    batch_size: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (true label indices, predicted label indices) for a catalog slice."""
    import torch
    from torch.utils.data import DataLoader

    device = resolve_device(config.device)
    dataset = CatalogImageDataset(frame, classes, build_transforms(config, train=False))
    loader = DataLoader(dataset, batch_size=batch_size or config.batch_size, shuffle=False)

    model = model.to(device).eval()
    y_true: list[int] = []
    y_pred: list[int] = []
    with torch.no_grad():
        for images, targets in loader:
            outputs = model(images.to(device))
            y_pred.extend(outputs.argmax(1).cpu().numpy().tolist())
            y_true.extend(torch.as_tensor(targets).numpy().tolist())
    return np.array(y_true), np.array(y_pred)


class CnnRecognizer(Recognizer):
    name = "cnn"

    def __init__(self, model: nn.Module, classes: list[str], config: CnnConfig | None = None):
        super().__init__(classes)
        self.config = config or CnnConfig()
        self.device = resolve_device(self.config.device)
        self.model = model.to(self.device).eval()
        self._transform = build_transforms(self.config, train=False)

    def predict_image(self, image_bgr: np.ndarray, top_k: int = 3) -> Prediction:
        import torch
        from PIL import Image

        rgb = np.asarray(image_bgr)[:, :, ::-1]
        tensor = self._transform(Image.fromarray(rgb)).unsqueeze(0).to(self.device)
        with _Timer() as timer, torch.no_grad():
            logits = self.model(tensor)
            probabilities = torch.softmax(logits, dim=1)[0].cpu().numpy()
        index = int(np.argmax(probabilities))
        return Prediction(
            label=self.classes[index],
            confidence=float(probabilities[index]),
            latency_ms=timer.elapsed_ms,
            scores=self.top_k(probabilities, top_k),
        )

    def save(self, directory: str | Path) -> Path:
        import torch

        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), directory / WEIGHTS_FILE)
        (directory / META_FILE).write_text(
            json.dumps(
                {
                    "backend": self.name,
                    "classes": self.classes,
                    "cnn_config": json.loads(self.config.model_dump_json()),
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        return directory

    @classmethod
    def load(cls, directory: str | Path) -> CnnRecognizer:
        import torch

        directory = Path(directory)
        meta = json.loads((directory / META_FILE).read_text(encoding="utf-8"))
        config = CnnConfig.model_validate(meta["cnn_config"])
        model = build_model(len(meta["classes"]), config.image_size)
        state = torch.load(
            directory / WEIGHTS_FILE, map_location=resolve_device(config.device), weights_only=True
        )
        model.load_state_dict(state)
        return cls(model=model, classes=meta["classes"], config=config)


def iter_batches(items: Iterable, size: int):  # pragma: no cover - helper
    batch: list = []
    for item in items:
        batch.append(item)
        if len(batch) == size:
            yield batch
            batch = []
    if batch:
        yield batch


__all__ = [
    "CatalogImageDataset",
    "CnnRecognizer",
    "build_model",
    "build_transforms",
    "predict_frame_batch",
    "resolve_device",
    "train_cnn",
]
