"""MiniMax H3 native loader integration.

State-dict routing and component construction for H3 checkpoints, moved out of the
extension's runtime hooks: the loader calls these functions directly for the
`MiniMax-H3` guess.
"""

import logging

import torch
from transformers.modeling_utils import no_init_weights

from backend import memory_management
from backend.nn import krea
from backend.nn.llm import llama
from backend.operations import using_forge_operations
from backend.state_dict import load_state_dict

from . import filebacked, gguf_vision, model, release, vae
from .islands import fp32_islands, restore_fp32
from .text_encoder import Qwen3VL32B
from .transformer import MiniMaxH3Model

logger = logging.getLogger("MiniMax H3")

TE_KEYS = ("visual.deepstack_merger_list.0.norm.weight", "model.layers.49.self_attn.q_proj.weight")
TE_PREFIX = "qwen3vl_32b.transformer."

MISSING_TE = (
    "MiniMax H3 needs its Qwen3-VL 32B text encoder: select a qwen3vl_32b_minimax_h3 file under VAE / Text Encoder"
)


def is_h3(guess) -> bool:
    return isinstance(guess, model.MiniMaxH3)


def _is_h3_text_encoder(asd: dict) -> bool:
    return all(k in asd for k in TE_KEYS)


def _replace(sd: dict, prefix: str, asd: dict) -> None:
    for k in [k for k in sd if k.startswith(prefix)]:
        del sd[k]
    for k, v in asd.items():
        sd[f"{prefix}{k}"] = v


def replace_state_dict(sd: dict, asd: dict, guess, path) -> None:
    """Route one additional module into the main state dict, in place."""
    if _is_h3_text_encoder(asd):
        _replace(sd, f"{guess.text_encoder_key_prefix[0]}{TE_PREFIX}", asd)
    elif vae.is_video_vae(asd):
        _replace(sd, f"{guess.vae_key_prefix[0]}video.", vae.convert_video_vae(asd))
    elif vae.is_audio_vae(asd):
        _replace(sd, f"{guess.vae_key_prefix[0]}audio.", vae.convert_audio_vae(asd))


def load_vae(state_dict: dict):
    """Build the video+audio VAE pair from the merged `vae.` state dict."""
    if (
        not isinstance(state_dict, dict)
        or not any(k.startswith("video.") for k in state_dict)
        or not any(k.startswith("audio.") for k in state_dict)
    ):
        raise ValueError(vae.MISSING_VAE)
    layers = vae.video_layers(state_dict)
    with no_init_weights():
        video = None
        if vae.is_quantized(state_dict, "video."):
            # Kijai's int8 video VAE: Forge's mixed-precision operations read each layer's comfy_quant;
            # the audio VAE stays on plain operations in fp32
            with using_forge_operations(
                device=memory_management.cpu, dtype=torch.float32, manual_cast_enabled=True, extra_dtype={"mixed_ops": True, "TE": False}
            ):
                video = vae.MiniMaxH3VideoVAE(num_layers=layers)
        with using_forge_operations(device=memory_management.cpu, dtype=torch.float32, extra_dtype="vae"):
            pair = vae.AutoencoderMiniMaxH3(video_layers=layers, video=video)
    load_state_dict(pair, state_dict, log_name="MiniMax H3 VAE")
    return pair


def text_encoder_class():
    """The 32B Qwen3-VL class to build instead of Forge's 4B/8B one."""
    return Qwen3VL32B


def before_text_encoder_build(state_dict: dict) -> bool:
    """Convert a GGUF encoder's vision tower into plain parameters; returns whether it was GGUF."""
    gguf = gguf_vision.is_gguf(state_dict)
    if gguf:
        from backend.loader_gguf import dequantize

        gguf_vision.plain_vision(state_dict, dequantize)
    return gguf


def after_text_encoder_build(model, gguf: bool) -> None:
    if gguf:
        gguf_vision.cast_to_input(model.visual)


def transformer_class():
    from .transformer import MiniMaxH3Model

    return MiniMaxH3Model


def transformer_islands(state_dict: dict):
    return fp32_islands(state_dict, curve="adaln_t_table" in state_dict)


def restore_transformer_islands(model, islands) -> None:
    restore_fp32(model, islands)


def keep_file_backed(name: str, module, storages) -> None:
    try:
        backed, total = filebacked.keep_file_backed(module, storages)
    except Exception as e:
        logger.warning(f"[MiniMax H3] {name}: weights will be copied to system RAM when they leave VRAM: {e}")
        return
    print(f"[MiniMax H3] {name}: {backed / 2**30:.1f} of {total / 2**30:.1f} GiB stay file-backed in system RAM")


def file_storages(state_dict: dict):
    return filebacked.file_storages(state_dict)


def release_replaced(loaded_models, active_model) -> int:
    try:
        return release.release_replaced(loaded_models, active_model, release.h3_module_types())
    except Exception as e:
        logger.warning(f"[MiniMax H3] could not release the replaced H3 model early: {e}")
        return 0


def reload_instead_of_unload(need_global_unload: bool, active_model) -> bool:
    return release.reload_instead_of_unload(need_global_unload, active_model)
