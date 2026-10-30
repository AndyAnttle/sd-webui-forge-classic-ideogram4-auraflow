from backend import memory_management
from backend.nn.base import ModuleDict, ObjectDict
from backend.patcher.base import ModelPatcher


class JointTextEncoder(ModuleDict):
    pass


class CLIP:
    def __init__(self, model_dict={}, tokenizer_dict={}, no_init=False):
        if no_init:
            return

        load_device = memory_management.text_encoder_device()
        offload_device = memory_management.text_encoder_offload_device()

        self.cond_stage_model = JointTextEncoder(model_dict)
        self.tokenizer = ObjectDict(tokenizer_dict)
        self.patcher = ModelPatcher(self.cond_stage_model, load_device=load_device, offload_device=offload_device)

        try:
            memory_management.logger.info(
                "Text encoder placement: load=%s, offload=%s, size=%.2f GiB, total=%.2f GiB",
                load_device, self.patcher.offload_device, self.patcher.model_size() / 2**30,
                memory_management.get_total_memory(load_device) / 2**30)
        except Exception:
            pass

        # A text encoder explicitly placed on its own GPU stays resident there (see
        # memory_management.text_encoder_offload_device), but only while it leaves room to work:
        # a 32B encoder that fills the card cannot allocate the transient buffers of the next
        # prompt encode (MiniMax H3: 19.5 of 24 GB, OOM on the third prompt). Anything that takes
        # more than ~60% of the device goes back to CPU offloading instead.
        if offload_device == load_device and load_device.type == "cuda":
            try:
                if self.patcher.model_size() > memory_management.get_total_memory(load_device) * 0.6:
                    self.patcher.offload_device = memory_management.cpu
            except Exception:
                pass

    def clone(self):
        n = CLIP(no_init=True)
        n.patcher = self.patcher.clone()
        n.cond_stage_model = self.cond_stage_model
        n.tokenizer = self.tokenizer
        return n

    def add_patches(self, *arg, **kwargs):
        return self.patcher.add_patches(*arg, **kwargs)
