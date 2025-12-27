#!/usr/bin/env python

# Copyright 2024 Tony Z. Zhao and The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Action Chunking Transformer Policy

As per Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware (https://huggingface.co/papers/2304.13705).
The majority of changes here involve removing unused code, unifying naming, and adding helpful comments.
"""

from collections import deque

import torch
import torch.nn.functional as F  # noqa: N812
import torchvision
from lerobot.policies.act.modeling_act import ACT, ACTTemporalEnsembler
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.utils.constants import ACTION, OBS_IMAGES
from torch import Tensor, nn
from torchvision.models._utils import IntermediateLayerGetter

from lerobot_policy_act_ttt.ttt import TTTVisionBackbone

# from lerobot.policies.act.configuration_act import ACTConfig
from .configuration_act_ttt import ACT_TTTConfig


def replace_batchnorm_with_groupnorm(model, num_groups=32):
    """
    Recursively replace all BatchNorm2d layers with GroupNorm.

    Args:
        model: The model to modify
        num_groups: Number of groups for GroupNorm (default: 32)
    """
    for name, module in model.named_children():
        if isinstance(module, nn.BatchNorm2d):
            # Get the number of channels
            num_channels = module.num_features

            # Create GroupNorm layer
            # Ensure num_groups divides num_channels
            groups = min(num_groups, num_channels)
            while num_channels % groups != 0:
                groups -= 1

            new_layer = nn.GroupNorm(
                num_groups=groups,
                num_channels=num_channels,
                eps=module.eps,
                affine=module.affine,
            )

            # Copy weights if affine=True
            if module.affine:
                new_layer.weight.data = module.weight.data.clone()
                new_layer.bias.data = module.bias.data.clone()

            # Replace the layer
            setattr(model, name, new_layer)
        else:
            # Recursively apply to child modules
            replace_batchnorm_with_groupnorm(module, num_groups)

    return model


class ACT_TTTPolicy(PreTrainedPolicy):
    """
    Action Chunking Transformer Policy as per Learning Fine-Grained Bimanual Manipulation with Low-Cost
    Hardware (paper: https://huggingface.co/papers/2304.13705, code: https://github.com/tonyzhaozh/act)

    Extended with Test Time Training (TTT) capabilities. See https://yueatsprograms.github.io/ttt/home.html
    for paper and code of the original method.
    """

    config_class = ACT_TTTConfig
    name = "act_ttt"

    def __init__(
        self,
        config: ACT_TTTConfig,
        **kwargs,
    ):
        """
        Args:
            config: Policy configuration class instance or None, in which case the default instantiation of
                    the configuration class is used.
        """
        super().__init__(config)
        config.validate_features()
        self.config = config

        self.model = ACT(self.config)
        backbone_model = getattr(torchvision.models, config.vision_backbone)(
            replace_stride_with_dilation=[
                False,
                False,
                config.replace_final_stride_with_dilation,
            ],
            weights=config.pretrained_backbone_weights,
        )
        backbone_model = replace_batchnorm_with_groupnorm(backbone_model)

        self.model.backbone = IntermediateLayerGetter(
            backbone_model, return_layers={"layer4": "feature_map"}
        )

        if config.temporal_ensemble_coeff is not None:
            self.temporal_ensembler = ACTTemporalEnsembler(
                config.temporal_ensemble_coeff, config.chunk_size
            )

        self.model.backbone = TTTVisionBackbone(
            self.model.backbone,
            task=config.ttt_transform,
            steps=config.ttt_steps,
            n_samples=config.ttt_n_samples,
            ss_head_latent_dim=config.ttt_ss_head_latent_dim,
            ss_head_layers=config.ttt_ss_head_layers,
            optimizer=config.ttt_optimizer,
            optimizer_config=config.ttt_optimizer_config,
        )

        self.ttt_weight = config.joint_loss_weight
        self.ttt_enabled = True

        self.reset()

    def get_optim_params(self) -> dict:
        # TODO(aliberts, rcadene): As of now, lr_backbone == lr
        # Should we remove this and just `return self.parameters()`?
        return [
            {
                "params": [
                    p
                    for n, p in self.named_parameters()
                    if not n.startswith("model.backbone") and p.requires_grad
                ]
            },
            {
                "params": [
                    p
                    for n, p in self.named_parameters()
                    if n.startswith("model.backbone") and p.requires_grad
                ],
                "lr": self.config.optimizer_lr_backbone,
            },
        ]

    def reset(self):
        """This should be called whenever the environment is reset."""
        if self.config.temporal_ensemble_coeff is not None:
            self.temporal_ensembler.reset()
        else:
            self._action_queue = deque([], maxlen=self.config.n_action_steps)

    @torch.no_grad()
    def select_action(self, batch: dict[str, Tensor]) -> Tensor:
        """Select a single action given environment observations.

        This method wraps `select_actions` in order to return one action at a time for execution in the
        environment. It works by managing the actions in a queue and only calling `select_actions` when the
        queue is empty.
        """
        self.eval()  # keeping the policy in eval mode as it could be set to train mode while queue is consumed

        if self.config.temporal_ensemble_coeff is not None:
            actions = self.predict_action_chunk(batch)
            action = self.temporal_ensembler.update(actions)
            return action

        # Action queue logic for n_action_steps > 1. When the action_queue is depleted, populate it by
        # querying the policy.
        if len(self._action_queue) == 0:
            actions = self.predict_action_chunk(batch)[:, : self.config.n_action_steps]

            # `self.model.forward` returns a (batch_size, n_action_steps, action_dim) tensor, but the queue
            # effectively has shape (n_action_steps, batch_size, *), hence the transpose.
            self._action_queue.extend(actions.transpose(0, 1))
        return self._action_queue.popleft()

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, Tensor]) -> Tensor:
        """Predict a chunk of actions given environment observations."""
        self.eval()

        if self.config.image_features:
            batch = dict(
                batch
            )  # shallow copy so that adding a key doesn't modify the original
            batch[OBS_IMAGES] = [batch[key] for key in self.config.image_features]
            if self.ttt_enabled:
                for img in batch[OBS_IMAGES]:
                    self.model.backbone.test_time_training(img)
                # Disable TTT during action selection
        actions = self.model(batch)[0]
        return actions

    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict]:
        """Run the batch through the model and compute the loss for training or validation."""
        if self.config.image_features:
            batch = dict(
                batch
            )  # shallow copy so that adding a key doesn't modify the original
            batch[OBS_IMAGES] = [batch[key] for key in self.config.image_features]

        actions_hat, (mu_hat, log_sigma_x2_hat) = self.model(batch)

        l1_loss = (
            F.l1_loss(batch[ACTION], actions_hat, reduction="none")
            * ~batch["action_is_pad"].unsqueeze(-1)
        ).mean()

        loss_dict = {"l1_loss": l1_loss.item()}
        loss = l1_loss

        if self.config.image_features:
            # Add self-supervised loss
            img_batch = torch.cat(batch[OBS_IMAGES], dim=0)
            ss_loss = self.model.backbone.ss_loss(img_batch, n=1)
            loss_dict["ss_loss"] = ss_loss.item()
            loss += self.ttt_weight * ss_loss

        if self.config.use_vae:
            # Calculate Dₖₗ(latent_pdf || standard_normal). Note: After computing the KL-divergence for
            # each dimension independently, we sum over the latent dimension to get the total
            # KL-divergence per batch element, then take the mean over the batch.
            # (See App. B of https://huggingface.co/papers/1312.6114 for more details).
            mean_kld = (
                (
                    -0.5
                    * (1 + log_sigma_x2_hat - mu_hat.pow(2) - (log_sigma_x2_hat).exp())
                )
                .sum(-1)
                .mean()
            )
            loss_dict["kld_loss"] = mean_kld.item()
            loss = loss + mean_kld * self.config.kl_weight

        return loss, loss_dict
