"""MiniMax H3 registration helpers: the guess lives in huggingface_guess.model_list, detection is below."""

from huggingface_guess.latent import MiniMaxH3Latent  # noqa: F401  (re-exported for the engine)
from huggingface_guess.model_list import MiniMaxH3  # noqa: F401  (re-exported for the engine)

from . import detect as detection

VIDEO_SHIFT = 12.0
AUDIO_SHIFT = 3.0


def detect(state_dict: dict, key_prefix: str) -> dict | None:
    return detection.detect(state_dict, key_prefix)
