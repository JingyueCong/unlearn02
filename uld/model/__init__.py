from peft import PeftModel
from transformers import AutoConfig

from .contrastllm import ContrastLLM
from .dualcontrastllm import DualContrastLLM
from .offsetllm import create_offset_model
from .utils import *
from ..utils import NameTimer

TRAIN_INIT_FUNCS = {
    "base": create_full_model,
    "uld": create_peft_model,
    "offset": create_offset_model,
}

def eval_create_base_model(base_model_config, model_mode_config, ckpt_path, device):
    with NameTimer("Loading Base model"):
        if os.path.exists(os.path.join(ckpt_path, 'adapter_config.json')):
            #! A lora model
            if os.path.exists(os.path.join(ckpt_path, '../fullmodel')):
                # small assistant
                base_path = os.path.join(ckpt_path, '../fullmodel')
            else:
                base_path = base_model_config.model_path
            model = AutoModelForCausalLM.from_pretrained(
                base_path, torch_dtype=torch.bfloat16
            ).to(device)
            peftmod = PeftModel.from_pretrained(
                model, ckpt_path, torch_dtype=torch.bfloat16
            )
            peftmod = peftmod.merge_and_unload()
            peftmod = peftmod.to(device)
            return peftmod 
        else:
            # Base only
            model = AutoModelForCausalLM.from_pretrained(
                ckpt_path, torch_dtype=torch.bfloat16
            ).to(device)
            return model


def eval_create_uld_model(base_model_config, model_mode_config, ckpt_path, device):
    with NameTimer("Loading ULD model"):
        basellm = AutoModelForCausalLM.from_pretrained(
            base_model_config.model_path, torch_dtype=torch.bfloat16
        ).to(device)
        with NameTimer("Loading assistant"):
            small_full_path = os.path.join(ckpt_path, '../fullmodel')
            assistant = AutoModelForCausalLM.from_pretrained(
                small_full_path, torch_dtype=torch.bfloat16
            ).to(device)
            peftmod = PeftModel.from_pretrained(
                assistant, ckpt_path, torch_dtype=torch.bfloat16
            )
            peftmod = peftmod.merge_and_unload()
            peftmod = peftmod.to(device)
        
        model = ContrastLLM(
            basellm, peftmod, 
            weight=model_mode_config.weight, 
            top_logit_filter=model_mode_config.top_logit_filter,
        ) 
        return model

def eval_create_offset_model(base_model_config, model_mode_config, ckpt_path, device):
    with NameTimer("Loading Offset model"):
        config = AutoConfig.from_pretrained(ckpt_path)
        if hasattr(config, 'is_offset') and config.is_offset:
            if hasattr(config, 'weight'):
                weight = config.weight
            else:
                weight = 1.0
            base_name = config.base_model_name
            model = create_offset_model(
                base_name, 
                device=device, 
                base_assist_path=config.base_assist_path, 
                weight=weight, 
                new_assist_path=ckpt_path
            )
            return model

def _load_assistant(base_model_path, assist_ckpt_path, device):
    """Load a small ULD assistant: full small LLM saved next to the LoRA adapter,
    then merge the LoRA weights."""
    small_full_path = os.path.join(assist_ckpt_path, '../fullmodel')
    assistant = AutoModelForCausalLM.from_pretrained(
        small_full_path, torch_dtype=torch.bfloat16
    ).to(device)
    peftmod = PeftModel.from_pretrained(
        assistant, assist_ckpt_path, torch_dtype=torch.bfloat16
    )
    peftmod = peftmod.merge_and_unload()
    peftmod = peftmod.to(device)
    return peftmod


def eval_create_dual_uld_model(base_model_config, model_mode_config, ckpt_path, device):
    with NameTimer("Loading Dual-ULD model"):
        basellm = AutoModelForCausalLM.from_pretrained(
            base_model_config.model_path, torch_dtype=torch.bfloat16
        ).to(device)
        # Two independent assistants. Accept either explicit a1/a2 paths in the
        # model_mode config, or fall back to ckpt_path as a1 path for convenience.
        a1_path = model_mode_config.get('a1_ckpt_path') or ckpt_path
        a2_path = model_mode_config.get('a2_ckpt_path')
        if a2_path is None:
            raise ValueError("dual_uld requires model_mode.a2_ckpt_path")

        with NameTimer("Loading assistant A1"):
            a1 = _load_assistant(base_model_config.model_path, a1_path, device)
        with NameTimer("Loading assistant A2"):
            a2 = _load_assistant(base_model_config.model_path, a2_path, device)

        model = DualContrastLLM(
            basellm, a1, a2,
            weight_a1=model_mode_config.weight_a1,
            weight_a2=model_mode_config.weight_a2,
            top_logit_filter=model_mode_config.top_logit_filter,
        )
        return model


TRAIN_INIT_FUNCS = {
    "base": create_full_model,
    "uld": create_peft_model,
    "offset": create_offset_model,
    # dual_uld is eval-only: train A1 and A2 separately with model_mode=uld.
}

EVAL_INIT_FUNCS = {
    "base": eval_create_base_model,
    "uld": eval_create_uld_model,
    "offset": eval_create_offset_model,
    "dual_uld": eval_create_dual_uld_model,
}