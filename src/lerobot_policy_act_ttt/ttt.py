from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass

from typing import Callable

import torch
import torch.nn.functional as F
from torch import nn
from torchvision.models._utils import IntermediateLayerGetter


def flip_transform(image):
    # Get a random label
    label = torch.randint(0, 5, (1,)).long()
    transformed_image = None

    if label.item() == 0:
        transformed_image = image  # No transformation
    elif label.item() == 1:
        transformed_image = torch.flip(image, dims=[2])  # Horizontal flip
    elif label.item() == 2:
        transformed_image = torch.flip(image, dims=[1])  # Vertical flip
    elif label.item() == 3:
        transformed_image = torch.flip(image, dims=[1, 2])  # Both flips
    elif label.item() == 4:  # 180 degree rotation
        transformed_image = torch.rot90(image, k=2, dims=[1, 2])

    return label, transformed_image


def perm_to_index(perm):
    """Convert a permutation to its lexicographic index (Lehmer code)."""
    n = len(perm)
    perm = perm.tolist()
    index = 0
    factorial = 1
    for i in range(1, n):
        factorial *= i
    for i in range(n - 1):
        # Count elements smaller than perm[i] that appear after position i
        count = sum(1 for j in range(i + 1, n) if perm[j] < perm[i])
        index += count * factorial
        factorial //= (n - 1 - i) if (n - 1 - i) > 0 else 1
    return index


def jigsaw_task(img, border_ratio=0.1):
    grid_x, grid_y = 2, 2  # 2x2 grid

    C, H, W = img.shape  # img is (C, H, W) - single image without batch dimension
    tile_h, tile_w = H // grid_y, W // grid_x
    tiles = []
    for i in range(grid_y):
        for j in range(grid_x):
            tile = img[:, i * tile_h : (i + 1) * tile_h, j * tile_w : (j + 1) * tile_w]
            tiles.append(tile)
    # Generate a random permutation of the tiles
    perm = torch.randperm(len(tiles))
    permuted_tiles = [tiles[i] for i in perm]
    # Create the transformed image
    transformed_image = torch.zeros_like(img)

    # Calculate border size in pixels
    border_h = int(tile_h * border_ratio)
    border_w = int(tile_w * border_ratio)

    for idx, tile in enumerate(permuted_tiles):
        i = idx // grid_x
        j = idx % grid_x

        # Clone the tile and black out borders to prevent corner-based shortcuts
        tile_with_border = tile.clone()
        # Top border
        tile_with_border[:, :border_h, :] = 0
        # Bottom border
        tile_with_border[:, -border_h:, :] = 0
        # Left border
        tile_with_border[:, :, :border_w] = 0
        # Right border
        tile_with_border[:, :, -border_w:] = 0

        transformed_image[
            :, i * tile_h : (i + 1) * tile_h, j * tile_w : (j + 1) * tile_w
        ] = tile_with_border
    # Convert permutation to a single index (0-23 for 4! permutations)
    label = torch.tensor(perm_to_index(perm), dtype=torch.long)
    return label, transformed_image


def color_permutation_transform(image):
    # Generate a random permutation of the color channels
    perm = torch.randperm(3)
    transformed_image = image[perm, :, :]
    # Convert permutation to a single index (0-5 for 3! permutations)
    label = torch.tensor(perm_to_index(perm), dtype=torch.long)
    return label, transformed_image


def get_ttt_transform(transform_type: str) -> tuple[Callable, int]:
    if transform_type == "flip":
        return flip_transform, 5
    elif transform_type == "jigsaw":
        return jigsaw_task, 24  # 2x2 grid has 4! = 24 permutations
    elif transform_type == "color":
        return color_permutation_transform, 6  # 3! = 6 permutations

    raise ValueError(f"Unknown transform type: {transform_type}")


def get_optimizer(optimizer_name: str, parameters, optimizer_config: dict):
    cls = getattr(torch.optim, optimizer_name)

    return cls(parameters, **optimizer_config)


class TTTVisionBackbone(nn.Module):
    """Test-Time Training module for fine-tuning the vision backbone during inference.

    - has a list of self-supervised tasks
    - the backbone model is fine-tuned using a self-supervised loss on the transformed images

    - during training only the self-supervised head is updated
    - during inference the self-supervised heas is frozen and the backbone is updated

    Args:
        backbone_model: The vision backbone model to be fine-tuned. (resnet)
    """

    def __init__(
        self,
        backbone_model: IntermediateLayerGetter,
        task: str,
        steps: int,
        n_samples: int,
        ss_head_latent_dim: int,
        ss_head_layers: int,
        optimizer: str,
        optimizer_config: dict,
    ) -> None:
        super().__init__()
        self.backbone_model = backbone_model
        self.task, n_classes = get_ttt_transform(task)
        self.steps = steps
        self.n_samples = n_samples

        layers: list[nn.Module] = [
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
        ]
        out_size = 0
        # determine the output size of the backbone
        with torch.no_grad():
            dummy_input = torch.zeros(1, 3, 224, 224).to(
                next(self.backbone_model.parameters()).device
            )
            feature_map = self.backbone_model(dummy_input)["feature_map"]
            pooled = layers[0](feature_map)
            flattened = layers[1](pooled)
            feature_map = flattened
            out_size = feature_map.shape[1]

        if ss_head_layers > 0:
            for i in range(ss_head_layers - 1):
                layers.append(
                    nn.Linear(
                        ss_head_latent_dim if i > 0 else out_size,
                        ss_head_latent_dim,
                    )
                )
                layers.append(nn.ReLU())
            layers.append(nn.Linear(ss_head_latent_dim, n_classes))
        else:
            layers.append(nn.Linear(out_size, n_classes))

        self.ss_head = nn.Sequential(
            *layers,
        )
        optim_params = [
            {"params": p}
            for n, p in self.backbone_model.named_parameters()
            if n.startswith("layer1") or n.startswith("layer2")
        ]
        self.optim = get_optimizer(optimizer, optim_params, optimizer_config)

    def forward(self, x):
        return self.backbone_model(x)

    def ss_loss(self, batch, n=10):
        imgs, labels = self._get_samples(batch, n)
        preds = self.ss_head(self.backbone_model(imgs)["feature_map"])
        loss = F.cross_entropy(preds, labels.to(preds.device))
        return loss

    def test_time_training(self, batch):
        # Enable gradients even if called from a no_grad context (e.g., select_action)
        with torch.enable_grad():
            self.backbone_model.train()
            self.ss_head.eval()

            for _ in range(self.steps):
                loss = self.ss_loss(batch, n=self.n_samples)
                self.optim.zero_grad()
                loss.backward()
                self.optim.step()

    def _get_samples(self, batch, n=10):
        images = batch

        # Ensure batch dimension exists (handle single frame case)
        if images.dim() == 3:
            images = images.unsqueeze(0)

        batch_size = images.shape[0]

        transformed_images = []
        labels = []
        for i in range(batch_size):
            for _ in range(n):
                lbl, img_transformed = self.task(images[i])
                transformed_images.append(img_transformed.unsqueeze(0))
                labels.append(lbl)
        transformed_images = torch.cat(transformed_images, dim=0)
        labels = torch.stack(labels).squeeze()

        # Ensure labels has at least 1 dimension for cross_entropy
        if labels.dim() == 0:
            labels = labels.unsqueeze(0)

        return transformed_images, labels
