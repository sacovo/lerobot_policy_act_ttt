"""Custom policy package for LeRobot."""

try:
    import lerobot  # noqa: F401
except ImportError:
    raise ImportError(
        "lerobot is not installed. Please install lerobot to use this policy package."
    )

from .configuration_act_ttt import ACT_TTTConfig
from .modeling_act_ttt import ACT_TTTPolicy
from .processor_act_ttt import make_act_ttt_pre_post_processors

__all__ = [
    "ACT_TTTConfig",
    "ACT_TTTPolicy",
    "make_act_ttt_pre_post_processors",
]
