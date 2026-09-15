"""Explainability for the CNN backend: Grad-CAM and first-layer feature maps.

Useful beyond the pretty pictures: on this dataset a model can reach a very high
score while attending to the background rather than the hand, and Grad-CAM is
the cheapest way to catch it.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:  # pragma: no cover
    import torch
    from torch import nn


class GradCAM:
    """Gradient-weighted class activation map for one convolutional layer."""

    def __init__(self, model: nn.Module, target_layer: nn.Module) -> None:
        self.model = model
        self.activations: torch.Tensor | None = None
        self.gradients: torch.Tensor | None = None
        # register_full_backward_hook, not the deprecated register_backward_hook:
        # the partial hook silently drops gradients on multi-node forwards.
        self._handles = [
            target_layer.register_forward_hook(self._save_activation),
            target_layer.register_full_backward_hook(self._save_gradient),
        ]

    def _save_activation(self, module, inputs, output) -> None:  # noqa: ARG002
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output) -> None:  # noqa: ARG002
        self.gradients = grad_output[0].detach()

    def close(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles = []

    def __enter__(self) -> GradCAM:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    def generate(self, image: torch.Tensor, class_index: int | None = None) -> np.ndarray:
        import torch

        self.model.zero_grad(set_to_none=True)
        output = self.model(image)
        if class_index is None:
            class_index = int(output.argmax(dim=1).item())
        output[:, class_index].sum().backward()

        if self.activations is None or self.gradients is None:  # pragma: no cover
            raise RuntimeError("no activation captured - is the target layer inside the model?")

        weights = self.gradients[0].mean(dim=(1, 2))
        cam = torch.relu((weights[:, None, None] * self.activations[0]).sum(0))
        cam = cam - cam.min()
        peak = float(cam.max())
        if peak > 1e-8:
            cam = cam / peak
        return cam.cpu().numpy()


def overlay_heatmap(image_chw: np.ndarray, cam: np.ndarray, alpha: float = 0.4) -> np.ndarray:
    """Blend a CAM over a CHW float image in [0, 1]. Returns an HWC RGB array."""
    import cv2

    image = np.transpose(image_chw, (1, 2, 0))
    cam_resized = cv2.resize(cam, (image.shape[1], image.shape[0]))
    heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB) / 255.0
    return np.clip((1 - alpha) * image + alpha * heatmap, 0, 1)


def save_image(image_hwc: np.ndarray, path: str | Path, title: str = "") -> Path | None:
    """Save one RGB image in [0, 1]. Skipped silently without matplotlib."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:  # pragma: no cover - optional dependency
        return None

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, axis = plt.subplots(figsize=(3, 3))
    axis.imshow(np.clip(image_hwc, 0, 1))
    axis.set_title(title, fontsize=8)
    axis.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def feature_maps(
    model: nn.Module, image: torch.Tensor, layer_indices=(0, 4, 8)
) -> list[np.ndarray]:
    """Activations of the selected conv layers for one image."""
    import torch

    outputs: list[np.ndarray] = []
    activation = image
    with torch.no_grad():
        for index, layer in enumerate(model.conv):
            activation = layer(activation)
            if index in layer_indices:
                outputs.append(activation[0].cpu().numpy())
    return outputs


def save_grid(
    maps: np.ndarray, path: str | Path, title: str = "", max_maps: int = 32
) -> Path | None:
    """Save a grid of single-channel maps. Skipped silently without matplotlib."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:  # pragma: no cover - optional dependency
        return None

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    count = min(len(maps), max_maps)
    columns = 8
    rows = int(np.ceil(count / columns))
    fig, axes = plt.subplots(rows, columns, figsize=(columns * 1.5, rows * 1.5))
    fig.suptitle(title)
    for index, axis in enumerate(np.atleast_1d(axes).ravel()):
        axis.axis("off")
        if index < count:
            axis.imshow(maps[index], cmap="gray")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


__all__ = ["GradCAM", "feature_maps", "overlay_heatmap", "save_grid", "save_image"]
