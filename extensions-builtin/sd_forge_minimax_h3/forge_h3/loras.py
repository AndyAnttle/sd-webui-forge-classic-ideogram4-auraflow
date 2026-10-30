"""MiniMax H3 LoRAs: list the ones the user may pick, check a file before Forge loads it, build the prompt tag.

The dropdown shows every LoRA whose file name contains "minimax" or "h3" (any folder under models/Lora), so short
names work. The check reads only the safetensors header: a real H3 LoRA either says so in its metadata
("base_model": "Comfy-Org/MiniMax-H3 ...") or carries the H3 attention keys (diffusion_model.blocks.N.attn.qkv_proj),
which SD, SDXL, Pony or Wan LoRAs do not have.
"""

from __future__ import annotations

import json
import os
import re
import struct
from pathlib import Path

from modules import paths, shared

EXTENSIONS = (".safetensors", ".ckpt", ".pt", ".bin")
NAME_HINTS = ("minimax",)
# "h3" as a word: h3_turbo, minimax-h3, but not the "h3" inside names like epoch35
H3_NAME = re.compile(r"(^|[^a-z0-9])h3(?![0-9])")
MAX_BLOCKS = 64
_BLOCK = re.compile(r"blocks[._](\d+)")


def lora_dir() -> Path:
    configured = getattr(shared.cmd_opts, "lora_dir", None)
    return Path(configured or os.path.join(paths.models_path, "Lora"))


def names() -> list[str]:
    """Every LoRA with minimax or h3 in its file name, as paths relative to models/Lora without the extension."""
    root = lora_dir()
    found = []
    if not root.is_dir():
        return found
    for path in root.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in EXTENSIONS:
            continue
        stem = path.stem.lower()
        if not any(hint in stem for hint in NAME_HINTS) and not H3_NAME.search(stem):
            continue
        found.append(path.relative_to(root).with_suffix("").as_posix())
    return sorted(found, key=str.lower)


def find(name: str) -> Path | None:
    if not name:
        return None
    root = lora_dir()
    for extension in ("",) + EXTENSIONS:
        path = root / (name + extension)
        if path.is_file():
            return path
    return None


def _header(path: Path) -> tuple[dict, dict]:
    with open(path, "rb") as file:
        size = struct.unpack("<Q", file.read(8))[0]
        if size <= 0 or size > 100 * 1024 * 1024:
            raise ValueError("unexpected safetensors header size")
        data = json.loads(file.read(size))
    metadata = data.pop("__metadata__", {}) or {}
    return data, metadata


def check(name: str) -> tuple[bool, str]:
    """(compatible, one-line reason). Reads the header only, never the weights."""
    path = find(name)
    if path is None:
        return False, "File not found, press the refresh button."
    if path.suffix.lower() != ".safetensors":
        return False, "Only .safetensors LoRAs can be checked."
    try:
        tensors, metadata = _header(path)
    except Exception as error:
        return False, f"Cannot read the header: {error}"
    base = str(metadata.get("base_model", ""))
    keys = " ".join(tensors)
    blocks = {int(match) for match in _BLOCK.findall(keys)}
    has_h3_keys = "attn.qkv_proj" in keys and bool(blocks) and max(blocks) < MAX_BLOCKS
    if "minimax" not in base.lower() and not has_h3_keys:
        return False, "Not a MiniMax H3 LoRA: no H3 keys and no MiniMax base model in the metadata."
    rank = metadata.get("training_rank")
    parts = [f"H3 LoRA, {len(blocks)} blocks" if blocks else "H3 LoRA"]
    if rank:
        parts.append(f"rank {rank}")
    return True, ", ".join(parts)


def prompt_name(name: str) -> str:
    """Forge indexes LoRAs by the bare file name (networks.process_network_files), folders are not part of the alias."""
    return Path(name).stem


def tag(name: str, weight: float) -> str:
    return f"<lora:{prompt_name(name)}:{float(weight):g}>"
