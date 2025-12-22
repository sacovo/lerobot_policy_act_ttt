"""Custom policy package for LeRobot."""

try:
    import lerobot  # noqa: F401
except ImportError:
    raise ImportError(
        "lerobot is not installed. Please install lerobot to use this policy package."
    )

from .configuration_act import ACTTTTConfig 
from .modeling_act import ACTTTTPolicy
from .processor_act import make_actttt_pre_post_processors

print("This is a test")

__all__ = [
    "ACTTTTConfig",
    "ACTTTTPolicy",
    "make_actttt_pre_post_processors",
]