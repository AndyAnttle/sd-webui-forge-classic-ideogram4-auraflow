# MiniMax H3 in Forge: how to use it

The engine, loader and preset are native (`backend/minimax_h3`); the panel lives in this built-in extension.
The **MiniMax H3** accordion is present in txt2img and img2img and starts collapsed.

## 1. Files

| Role | File | Where |
|---|---|---|
| Checkpoint (DiT) | `h3-nvfp4-10eros.safetensors` (NVFP4, 12.5 GB) | `models/Stable-diffusion/MiniMax-H3/` |
| Text encoder | `qwen3vl_32b-Q4_K_M.gguf` (14.6 GB) | `models/text_encoder/MiniMax-H3/` |
| Video VAE | `minimax_h3_video_vae_fp16.safetensors` (5.2 GB) | `models/VAE/MiniMax-H3/` |
| Audio VAE | `minimax_h3_audio_vae_fp32.safetensors` (0.6 GB) | `models/VAE/MiniMax-H3/` |
| Turbo LoRA (optional) | `h3_turbo_8step.safetensors` | `models/Lora/MiniMax-H3/` |
| Fun ControlNet Union 2.0 (optional) | `minimax_h3_fun_controlnet_union_2.0_pruned_int8_convrot.safetensors` (4.5 GB) | `models/ControlNet/` |

All files live in `V:\VidModels\MiniMAX-H3` and are linked into these folders.

## 2. UI order

1. Pick **minimax-h3** in the **UI Preset** dropdown at the top (or in Settings / Presets).
2. **Checkpoint**: `MiniMax-H3/h3-nvfp4-10eros` (or INT8-pruned, or the Q4_K_M GGUF).
3. **VAE / Text Encoder** (multi-select): pick three files, the text encoder plus the video VAE and the audio VAE. Forge has no separate VAE dropdown, the VAE is chosen here.
4. Size: 768x432 for 16:9, 512x288 for quick tests; both sides are multiples of 16.
5. **Frames**: the slider follows the 5 + 17n grid (5, 22, 39 ... 362), default 124 (5.17 s). For a test, 56 (2.33 s).
6. Sampler **Res Multistep**, Schedule **Simple**, Steps **20**, CFG **1.0**, Shift **12** (the Shift field).
7. Open the **MiniMax H3** accordion: Output `Video`, Include generated audio `on`, Audio shift `3.0`, Soundtrack `Generated`.
8. Generate. The clip plays in Forge's player and is saved to `output/videos/*.mp4` (h264 24 fps plus AAC 32 kHz stereo).

## 3. Modes

- **T2V**: attach nothing.
- **I2V, first frame**: put the picture into **img2img** and generate from there (denoising is not used).
- **I2V, last frame**: add one picture to the **ImageStitch Integrated** gallery in txt2img.
- **First and last frame**: a picture in img2img plus a picture in ImageStitch.
- **Ref2VA**: the img2img picture is `<Picture 1>`, the ImageStitch gallery holds the next pictures (up to 9 in all); video and audio references go into the **Reference video and audio** section of the panel.

## 4. Speed

- **Turbo LoRA**: add `<lora:h3_turbo_8step:1>` to the prompt, Steps 8, Shift 6 (12 when speech matters).
- **Acc 8-step LoRA**: Euler, Steps 8, Shift 12.
- **FastH3**: Steps 8, Shift 10, enable Sparse Attention Integrated.
- **Spectrum** works as usual: a speedup without losing the audio.

## 5. VAEs

- Forge sets the VAE in the **VAE / Text Encoder** multi-select, there is no separate field.
- H3 needs **two** VAEs, video and audio. Without either of them the panel shows an error and the generation does not start.
- The X/Y/Z `VAE` axis does not apply to H3: the model has its own VAE pair, and the hot-swap axis is meant for image models.
- Do not plug an SDXL or SD15 VAE into H3, only the H3 files.

## 6. Reproducing the verified result

- Model: `h3-nvfp4-10eros` (NVFP4).
- Text encoder: `qwen3vl_32b-Q4_K_M.gguf`.
- VAEs: `minimax_h3_video_vae_fp16` + `minimax_h3_audio_vae_fp32`.
- 512x288, Frames 56, Steps 20, Res Multistep, Simple, CFG 1, Shift 12, seed 555.
- Prompt: `A red paper boat drifting on a calm lake at sunset, gentle ripples, soft ambient music`.
- Expected: an MP4 in `output/videos`, about 45-50 s on a 3090 after the model is loaded (the first run is longer while the 12.5 GB checkpoint loads).

## 7. LoRA and speedups

- The **LoRA** row lists every file in `models/Lora` whose name contains `minimax` or `h3` (h3 as a separate word, so `epoch35` is not picked up). The refresh button rescans the folder after new files are added.
- On selection the header is checked: `base_model` metadata with MiniMax-H3, or the `diffusion_model.blocks.N.attn.qkv_proj` keys with no more than 64 blocks. If the file is not an H3 LoRA, the status says why and the LoRA is not applied.
- **LoRA weight** 0.5-1.5, 1.0 by default.
- **8-step turbo preset** is enabled once a valid LoRA is selected: Steps 8, Shift 6, CFG 1.0.
- The `<lora:name:weight>` tag is added to the prompt automatically and kept in the infotext, so the result can be reproduced by hand.
- Example: `h3_turbo_8step` at 768x768, 124 frames, 8 steps takes about 4-5 minutes instead of 10:45 at 20 steps.
- **Important:** for NVFP4 checkpoints set **Diffusion in Low Bits** to `Automatic (fp16 LoRA)`. With plain `Automatic` Forge merges the LoRA into the NVFP4 weights: a reload of about 2.5 minutes and a step about 6 times slower (12 s/step against 2.1 s/step with on-the-fly).

## 8. Memory and launch

- Keep the 32B text encoder on the second card: `CUDA_VISIBLE_DEVICES` with two 3090s and `--text-enc-device cuda:1`.
- DiT NVFP4 about 12.5 GB, video VAE 5.2 GB, audio VAE 0.6 GB: fits a 24 GB card with room to spare.
- INT8-pruned (21 GB) leaves almost no headroom on a 24 GB card and the second run may fail; prefer NVFP4.
- `--cuda-malloc` is recommended, as for the other large models.
