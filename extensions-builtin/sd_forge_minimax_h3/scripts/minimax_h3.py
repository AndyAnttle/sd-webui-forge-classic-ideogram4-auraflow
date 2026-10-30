"""MiniMax H3 panel: the engine, loader and preset are native (backend/minimax_h3)."""

import sys
import traceback
from pathlib import Path

ROOT = str(Path(__file__).resolve().parents[1])
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from modules import script_callbacks, scripts, shared

from backend.logging import setup_logger
from forge_h3 import integration, ui

import logging

setup_logger(logging.getLogger("forge_h3"))

ENABLED = False
try:
    from backend.minimax_h3 import loading  # noqa: F401

    ENABLED = True
except Exception:
    print("[MiniMax H3] the native backend is missing, the panel stays disabled:")
    traceback.print_exc()


class Script(scripts.Script):
    sorting_priority = 10
    create_group = False

    def title(self):
        return "MiniMax H3"

    def show(self, is_img2img):
        return scripts.AlwaysVisible if ENABLED else False

    def ui(self, is_img2img):
        self.panel = ui.Panel(is_img2img)
        self.infotext_fields = [(self.panel.audio_shift, "H3 Audio shift")]
        return self.panel.inputs

    def before_process(self, p, *args):
        lora = (None, 1.0)
        inputs = len(self.panel.inputs) if getattr(self, "panel", None) is not None else 0
        # the LoRA name and weight travel last, so API callers that send only the first three values keep working
        if inputs and len(args) >= inputs - 1:
            lora = (args[-2], args[-1])
            args = args[:-2]
        integration.script_before_process(p, *args, lora=lora)

    def process(self, p, *args):
        integration.process(p)

    def process_before_every_sampling(self, p, *args, **kwargs):
        integration.before_sampling(p, kwargs["noise"])

    def post_sample(self, p, ps, *args):
        integration.after_sampling(p)

    def postprocess(self, p, processed, *args):
        integration.postprocess(p, processed)


def settings():
    section = ("forge_h3", "MiniMax H3")
    shared.opts.add_option("h3_ffmpeg_path", shared.OptionInfo("", "H3 FFmpeg executable (empty = automatic)", section=section))


script_callbacks.on_ui_settings(settings)
# a rebuilt UI starts with no panels: the ones of a previous build can never bind again
script_callbacks.on_before_ui(ui.reset)
script_callbacks.on_after_component(ui.capture)
script_callbacks.on_ui_tabs(ui.bind_all)
script_callbacks.on_app_started(ui.check_bindings)
script_callbacks.on_script_unloaded(ui.reset)
