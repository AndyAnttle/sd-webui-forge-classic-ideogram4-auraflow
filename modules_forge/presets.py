from enum import Enum


class PresetArch(Enum):
    sd = 1  # SD1
    xl = 2  # SDXL
    flux = 3  # Flux.1
    klein = 4  # Flux.2
    qwen = 5  # Qwen-Image
    lumina = 6  # Lumina-Image-2.0
    zit = 7  # Z-Image-Turbo
    wan = 8  # Wan2.2
    anima = 9  # Anima
    ernie = 10  # Ernie-Image
    pid = 11  # PiD
    krea = 12  # Krea2
    qwen21 = 13  # Qwen-Image-2.1
    ideogram4 = 14  # Ideogram 4
    auraflow = 15  # AuraFlow
    moss = 16  # MOSS-Image
    h3 = 17  # MiniMax H3

    @staticmethod
    def choices() -> list[str]:
        return [preset.name for preset in PresetArch]


# the UI shows the name; the enum member stays "h3" so old option keys and code keep working
PresetArch.h3._name_ = "minimax-h3"


SAMPLERS = {
    PresetArch.sd: "Euler a",
    PresetArch.xl: "Euler a",
    PresetArch.flux: "Euler",
    PresetArch.klein: "Euler",
    PresetArch.qwen: "LCM",
    PresetArch.lumina: "Res Multistep",
    PresetArch.zit: "Euler",
    PresetArch.wan: "Euler",
    PresetArch.anima: "ER SDE",
    PresetArch.ernie: "Euler",
    PresetArch.pid: "LCM",
    PresetArch.krea: "Euler",
    PresetArch.qwen21: "Euler",
    PresetArch.ideogram4: "Euler",
    PresetArch.auraflow: "Euler",
    PresetArch.moss: "Euler",
    PresetArch.h3: "Res Multistep",
}

SCHEDULERS = {
    PresetArch.sd: "Automatic",
    PresetArch.xl: "Automatic",
    PresetArch.flux: "Beta",
    PresetArch.klein: "Beta",
    PresetArch.qwen: "Normal",
    PresetArch.lumina: "Simple",
    PresetArch.zit: "Beta",
    PresetArch.wan: "Simple",
    PresetArch.anima: "Beta",
    PresetArch.ernie: "Simple",
    PresetArch.pid: "Simple",
    PresetArch.krea: "Simple",
    PresetArch.qwen21: "Simple",
    PresetArch.ideogram4: "Simple",
    PresetArch.auraflow: "FlowMatchEulerDiscrete",
    PresetArch.moss: "Simple",
    PresetArch.h3: "simple",
}

STEPS = {
    PresetArch.sd: 32,
    PresetArch.xl: 24,
    PresetArch.flux: 20,
    PresetArch.klein: 4,
    PresetArch.qwen: 8,
    PresetArch.lumina: 32,
    PresetArch.zit: 9,
    PresetArch.wan: 4,
    PresetArch.anima: 32,
    PresetArch.ernie: 8,
    PresetArch.pid: 4,
    PresetArch.krea: 8,
    PresetArch.qwen21: 32,
    PresetArch.ideogram4: 20,
    # AuraFlow/Pony v7: official fal example uses 50 steps; Pony author page says
    # "at least 30 steps". 30 is the safe floor, 40-50 for finer detail.
    PresetArch.auraflow: 30,
    PresetArch.moss: 50,
    PresetArch.h3: 20,
}

CFG = {
    PresetArch.sd: 6.0,
    PresetArch.xl: 4.5,
    PresetArch.flux: 1.0,
    PresetArch.klein: 1.0,
    PresetArch.qwen: 1.0,
    PresetArch.lumina: 4.0,
    PresetArch.zit: 1.0,
    PresetArch.wan: 1.0,
    PresetArch.anima: 4.0,
    PresetArch.ernie: 1.0,
    PresetArch.pid: 1.0,
    PresetArch.krea: 1.0,
    PresetArch.qwen21: 2.5,
    # Ideogram4 is NOT guidance-distilled: official pipeline uses real CFG
    # (V4_DEFAULT_20: guidance 7.0, last 2 steps polish at 3.0). The engine
    # applies the polish automatically; UI CFG must be 7.0.
    PresetArch.ideogram4: 7.0,
    # AuraFlow is NOT guidance-distilled (unlike Z-Image): fal official example uses
    # guidance_scale=3.5; SimpleTuner AuraFlow guide says 3.5-4.0; Pony community runs
    # at 3.5-4.75. CFG 1.0 here would disable guidance entirely (negative prompts ignored).
    PresetArch.auraflow: 3.5,
    PresetArch.h3: 1.0,
    PresetArch.moss: 3.0,
}

DISTILL = {
    PresetArch.flux: 3.0,
}

SHIFT = {
    PresetArch.xl: -9.0,
    PresetArch.lumina: 6.0,
    PresetArch.zit: 9.0,
    PresetArch.wan: 5.0,
    PresetArch.anima: 3.0,
    PresetArch.ernie: 3.0,
    PresetArch.pid: -1.5,
    PresetArch.krea: -1.15,
    PresetArch.auraflow: 3.5,
    PresetArch.moss: 3.0,
    PresetArch.h3: 12.0,
}

FRAMES = {
    PresetArch.wan.name: 16,
    PresetArch.h3.name: 24,
}

# per-preset Frames slider geometry: (minimum, step, maximum). MiniMax H3 accepts 5 + 17n frames only
FRAME_GRID = {
    PresetArch.h3.name: (5, 17, 362),
}

# a sensible first clip when the stored value is still the stock batch default (1)
FRAME_DEFAULT = {
    PresetArch.h3.name: 124,
}


def frame_grid(arch: str) -> tuple[int, int, int]:
    if arch in FRAME_GRID:
        return FRAME_GRID[arch]
    fps = is_video(arch)
    return (1, fps if fps > 1 else 1, max(8, fps * 15 + 1))


def align_frames(arch: str, value) -> int:
    """Snap a stored Frames value onto the preset's grid (h3: 5 + 17n), so the slider opens on a valid clip."""
    minimum, step, maximum = frame_grid(arch)
    try:
        frames = int(value)
    except (TypeError, ValueError):
        frames = minimum
    if frames < minimum:
        frames = FRAME_DEFAULT.get(arch, minimum)
    frames = max(minimum, min(maximum, frames))
    if step > 1:
        frames = minimum + step * round((frames - minimum) / step)
        frames = max(minimum, min(maximum, frames))
    return frames


def use_distill(arch: str) -> bool:
    return arch in [preset.name for preset in DISTILL.keys()]


def use_shift(arch: str) -> bool:
    return arch in [preset.name for preset in SHIFT.keys()]


def is_video(arch: str) -> int:
    return FRAMES.get(arch, 1)


def register(options_templates: dict):
    from gradio import Dropdown, Slider

    from modules.options import OptionInfo, OptionRow, options_section
    from modules.shared_items import list_samplers, list_schedulers

    for arch in PresetArch:
        name = arch.name

        options_templates.update(
            options_section(
                (None, "Forge Hidden Options"),
                {
                    f"forge_checkpoint_{name}": OptionInfo(None),
                    f"forge_additional_modules_{name}": OptionInfo([]),
                    f"forge_unet_storage_dtype_{name}": OptionInfo("Automatic"),
                },
            )
        )

        sampler, scheduler = SAMPLERS[arch], SCHEDULERS[arch]

        options_templates.update(
            options_section(
                (f"ui_{name}", name.upper(), "presets"),
                {
                    f"{name}_t2i_ss1": OptionRow(),
                    f"{name}_t2i_sampler": OptionInfo(sampler, "txt2img Sampler", Dropdown, lambda: {"choices": [x.name for x in list_samplers()], "allow_custom_value": True}),
                    f"{name}_t2i_scheduler": OptionInfo(scheduler, "txt2img Scheduler", Dropdown, lambda: {"choices": list_schedulers(), "allow_custom_value": True}),
                    f"{name}_t2i_ss0": OptionRow(),
                    f"{name}_i2i_ss1": OptionRow(),
                    f"{name}_i2i_sampler": OptionInfo(sampler, "img2img Sampler", Dropdown, lambda: {"choices": [x.name for x in list_samplers()], "allow_custom_value": True}),
                    f"{name}_i2i_scheduler": OptionInfo(scheduler, "img2img Scheduler", Dropdown, lambda: {"choices": list_schedulers(), "allow_custom_value": True}),
                    f"{name}_i2i_ss0": OptionRow(),
                },
            )
        )

        step = STEPS[arch]

        options_templates.update(
            options_section(
                (f"ui_{name}", name.upper(), "presets"),
                {
                    f"{name}_steps1": OptionRow(),
                    f"{name}_t2i_step": OptionInfo(step, "txt2img Steps", Slider, {"minimum": 0, "maximum": 150, "step": 1}),
                    f"{name}_t2i_hr_step": OptionInfo(step, "txt2img Hires. Steps", Slider, {"minimum": 0, "maximum": 150, "step": 1}),
                    f"{name}_i2i_step": OptionInfo(step, "img2img Steps", Slider, {"minimum": 0, "maximum": 150, "step": 1}),
                    f"{name}_steps0": OptionRow(),
                },
            )
        )

        cfg = CFG[arch]

        options_templates.update(
            options_section(
                (f"ui_{name}", name.upper(), "presets"),
                {
                    f"{name}_cfg1": OptionRow(),
                    f"{name}_t2i_cfg": OptionInfo(cfg, "txt2img CFG", Slider, {"minimum": 0, "maximum": 24, "step": 0.5}),
                    f"{name}_t2i_hr_cfg": OptionInfo(cfg, "txt2img Hires. CFG", Slider, {"minimum": 0, "maximum": 24, "step": 0.5}),
                    f"{name}_i2i_cfg": OptionInfo(cfg, "img2img CFG", Slider, {"minimum": 0, "maximum": 24, "step": 0.5}),
                    f"{name}_cfg0": OptionRow(),
                },
            )
        )

        if (distill := DISTILL.get(arch, None)) is not None:
            options_templates.update(
                options_section(
                    (f"ui_{name}", name.upper(), "presets"),
                    {
                        f"{name}_dcfg1": OptionRow(),
                        f"{name}_t2i_dcfg": OptionInfo(distill, "txt2img Distilled CFG", Slider, {"minimum": 1, "maximum": 24, "step": 0.5}),
                        f"{name}_t2i_hr_dcfg": OptionInfo(distill, "txt2img Hires. Distilled CFG", Slider, {"minimum": 1, "maximum": 24, "step": 0.5}),
                        f"{name}_i2i_dcfg": OptionInfo(distill, "img2img Distilled CFG", Slider, {"minimum": 1, "maximum": 24, "step": 0.5}),
                        f"{name}_dcfg0": OptionRow(),
                    },
                )
            )

        if (shift := SHIFT.get(arch, None)) is not None:
            options_templates.update(
                options_section(
                    (f"ui_{name}", name.upper(), "presets"),
                    {
                        f"{name}_show_shift": OptionInfo((shift > 0.0), "Display Shift Slider"),
                        f"{name}_dcfg1": OptionRow(),
                        f"{name}_t2i_dcfg": OptionInfo(abs(shift), "txt2img Shift", Slider, {"minimum": 1, "maximum": 24, "step": 0.5}),
                        f"{name}_t2i_hr_dcfg": OptionInfo(abs(shift), "txt2img Hires. Shift", Slider, {"minimum": 1, "maximum": 24, "step": 0.5}),
                        f"{name}_i2i_dcfg": OptionInfo(abs(shift), "img2img Shift", Slider, {"minimum": 1, "maximum": 24, "step": 0.5}),
                        f"{name}_dcfg0": OptionRow(),
                    },
                )
            )

        if (fps := FRAMES.get(arch.name, 1)) > 1:
            options_templates.update(
                options_section(
                    (f"ui_{name}", name.upper(), "presets"),
                    {
                        f"{name}_batch1": OptionRow(),
                        f"{name}_t2i_batch_size": OptionInfo(1, "txt2img Frames", Slider, {"minimum": 1, "maximum": fps * 15 + 1, "step": fps}),
                        f"{name}_i2i_batch_size": OptionInfo(1, "img2img Frames", Slider, {"minimum": 1, "maximum": fps * 15 + 1, "step": fps}),
                        f"{name}_batch0": OptionRow(),
                    },
                )
            )
        else:
            options_templates.update(
                options_section(
                    (f"ui_{name}", name.upper(), "presets"),
                    {
                        f"{name}_batch1": OptionRow(),
                        f"{name}_t2i_batch_size": OptionInfo(1, "txt2img Batch Size", Slider, {"minimum": 1, "maximum": 8, "step": 1}),
                        f"{name}_i2i_batch_size": OptionInfo(1, "img2img Batch Size", Slider, {"minimum": 1, "maximum": 8, "step": 1}),
                        f"{name}_batch0": OptionRow(),
                    },
                )
            )

        options_templates.update(
            options_section(
                (f"ui_{name}", name.upper(), "presets"),
                {
                    f"{name}_t2i_dim1": OptionRow(),
                    f"{name}_t2i_width": OptionInfo(0, "txt2img Width", Slider, {"minimum": 0, "maximum": 2048, "step": 64}),
                    f"{name}_i2i_width": OptionInfo(0, "img2img Width", Slider, {"minimum": 0, "maximum": 2048, "step": 64}),
                    f"{name}_t2i_dim0": OptionRow(),
                    f"{name}_i2i_dim1": OptionRow(),
                    f"{name}_t2i_height": OptionInfo(0, "txt2img Height", Slider, {"minimum": 0, "maximum": 2048, "step": 64}),
                    f"{name}_i2i_height": OptionInfo(0, "img2img Height", Slider, {"minimum": 0, "maximum": 2048, "step": 64}),
                    f"{name}_i2i_dim0": OptionRow(),
                },
            )
        )
