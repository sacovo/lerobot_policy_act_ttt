from typing import cast

import torch
from lerobot.configs.policies import PreTrainedConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.utils.import_utils import register_third_party_plugins
from torchvision.transforms import functional as F

from lerobot_policy_act_ttt import ACT_TTTPolicy
from lerobot_policy_act_ttt.ttt import TTTVisionBackbone


def test_ttt_adaption():
    repo_id = "sancov/so101-ros-act-v0.2_ttt"
    ds_repo_id = "sancov/so101-ros-red-ring"

    register_third_party_plugins()
    dataset = LeRobotDataset(ds_repo_id)

    cfg = PreTrainedConfig.from_pretrained(repo_id, force_download=True)
    cfg.pretrained_path = repo_id  # ensure weights are loaded from hub

    print(cfg)

    # assert cfg.type == "act_ttt"

    policy = make_policy(cfg, ds_meta=dataset.meta)
    assert isinstance(policy, ACT_TTTPolicy)

    model = cast(TTTVisionBackbone, policy.model.backbone)
    assert isinstance(model, TTTVisionBackbone)

    pre, post = make_pre_post_processors(policy.config)

    frame = dataset[4550]
    frame = pre(frame)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy.to(device)
    for key in frame:
        if isinstance(frame[key], torch.Tensor):
            frame[key] = frame[key].to(device)

    # Images from preprocessor have shape (1, C, H, W) - squeeze batch dim and stack
    imgs = [frame[key].squeeze(0) for key in policy.config.image_features]

    # blur = GaussianBlur(kernel_size=5, sigma=1.0)
    # imgs_blurred = [blur(img) for img in imgs]

    # Stack images along batch dimension: (N_images, C, H, W)
    imgs = torch.stack(imgs, dim=0)
    imgs_darked = F.adjust_brightness(imgs, brightness_factor=0.5)

    model.enabled = False
    features_original = model(imgs)["feature_map"]
    features_darked = model(imgs_darked)["feature_map"]

    model.enabled = True

    # features_original_adapted = model(imgs)["feature_map"]
    features_darked_adapted = model(imgs_darked)["feature_map"]

    # features_blurred_adapted should be closer to features_original than features_blurred
    dist_before = torch.norm(features_original - features_darked)
    dist_after = torch.norm(features_original - features_darked_adapted)

    assert dist_after < dist_before
