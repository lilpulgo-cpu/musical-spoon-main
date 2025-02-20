__version__ = "2024.8"

__all__ = [
    "prepare_model_for_kbit_training",
    "xformers",
    "xformers_attention",
    "xformers_version",
    "__version__",
    "HAS_FLASH_ATTENTION",
    "HAS_FLASH_ATTENTION_SOFTCAPPING",
    "PRE_CHECK",
    "platform_system",
    "patch_tokenizer",
    "get_statistics",
    "Unsloth_Offloaded_Gradient_Checkpointer",
    "offload_to_disk",
    "offload_input_embeddings",
    "offload_output_embeddings",
    "is_bfloat16_supported",
    "unsloth_offloaded_gradient_checkpoint",
    "torch_compile_options",
    "patch_linear_scaling",
    "patch_llama_rope_scaling",
    "check_gpu_memory",  # Renamed function
    "create_boolean_mask",
    "torch_amp_custom_fwd",
    "torch_amp_custom_bwd",
]

import torch
from typing import Union, Optional, List, Any, Callable, Tuple
from platform import system as platform_system
platform_system = platform_system()
import numpy as np
import warnings, subprocess, re, inspect, psutil, os, math
from packaging.version import Version

warnings.filterwarnings(action = "ignore", category = UserWarning,    module = "torch")
warnings.filterwarnings(action = "ignore", category = UserWarning,    module = "huggingface_hub")
warnings.filterwarnings(action = "ignore", category = UserWarning,    module = "trl")
warnings.filterwarnings(action = "ignore", category = FutureWarning,  module = "huggingface_hub")
warnings.filterwarnings(action = "ignore", category = FutureWarning,  module = "xformers")
warnings.filterwarnings(action = "ignore", category = RuntimeWarning, module = "subprocess")
warnings.filterwarnings(action = "ignore", category = UserWarning,    module = "transformers")
warnings.filterwarnings(action = "ignore", category = FutureWarning,  module = "accelerate")
warnings.filterwarnings(action = "ignore", category = RuntimeWarning, module = "multiprocessing")
warnings.filterwarnings(action = "ignore", category = RuntimeWarning, module = "multiprocess")

import logging
logging.getLogger("transformers.tokenization_utils_base").setLevel(logging.CRITICAL+1)

import torch
from packaging.version import Version

torch_version = torch.__version__
is_cuda_available = torch.cuda.is_available()

torch_custom_fwd = None
torch_custom_bwd = None
torch_amp_custom_fwd = None
torch_amp_custom_bwd = None
custom_fwd = None
custom_bwd = None

def check_methods(device_type):
    global torch_custom_fwd, torch_custom_bwd, torch_amp_custom_fwd, torch_amp_custom_bwd, custom_fwd, custom_bwd
    
    try:
        if Version(torch_version) < Version("2.4.0"):
            # Pre 2.4.0 API
            try:
                torch_custom_fwd = getattr(torch, f"{device_type}.amp.custom_fwd", None)
                torch_custom_bwd = getattr(torch, f"{device_type}.amp.custom_bwd", None)
            except AttributeError:
                print(f"Advertencia: Los métodos 'custom_fwd' o 'custom_bwd' no están disponibles para {device_type} en esta versión.")
        else:
            # Post 2.4.0 API
            try:
                torch_custom_fwd = torch.amp.custom_fwd(device_type=device_type)
            except AttributeError as e:
                if "custom_fwd" in str(e):
                    print("Advertencia: module 'torch.amp' has no attribute 'custom_fwd'.")
                else:
                    print(f"Advertencia: Los métodos 'custom_fwd' no están disponibles para {device_type} en esta versión.")
                
            try:
                torch_custom_bwd = torch.amp.custom_bwd(device_type=device_type)
            except AttributeError as e:
                if "custom_bwd" in str(e):
                    print("Advertencia: module 'torch.amp' has no attribute 'custom_bwd'.")
                else:
                    print(f"Advertencia: Los métodos 'custom_bwd' no están disponibles para {device_type} en esta versión.")
            
            try:
                torch_amp_custom_fwd = torch.amp.custom_fwd(device_type=device_type)
            except AttributeError as e:
                if "custom_fwd" in str(e):
                    print("Advertencia: module 'torch.amp' has no attribute 'custom_fwd'.")
                else:
                    print(f"Advertencia: Los métodos AMP 'custom_fwd' no están disponibles para {device_type} en esta versión.")
                
            try:
                torch_amp_custom_bwd = torch.amp.custom_bwd(device_type=device_type)
            except AttributeError as e:
                if "custom_bwd" in str(e):
                    print("Advertencia: module 'torch.amp' has no attribute 'custom_bwd'.")
                else:
                    print(f"Advertencia: Los métodos AMP 'custom_bwd' no están disponibles para {device_type} en esta versión.")
            
            # Método universal (pre 2.4.0 y post 2.4.0)
            try:
                custom_fwd = torch.custom_fwd(device_type=device_type)
            except AttributeError as e:
                if "custom_fwd" in str(e):
                    print("Advertencia: module 'torch.fwd' has no attribute 'custom_fwd'.")
                else:
                    print(f"Advertencia: El método 'custom_fwd' no está disponible para {device_type} en esta versión.")
                
            try:
                custom_bwd = torch.custom_bwd(device_type=device_type)
            except AttributeError as e:
                if "custom_bwd" in str(e):
                    print("Advertencia: module 'torch.fwd' has no attribute 'custom_bwd'.")
                else:
                    print(f"Advertencia: El método 'custom_bwd' no está disponible para {device_type} en esta versión.")

    except AttributeError:
        print(f"Advertencia: La versión de PyTorch no soporta 'custom_fwd' o 'custom_bwd' para {device_type}.")

# Verifica los métodos para CUDA y CPU
if is_cuda_available:
    check_methods("cuda")
check_methods("cpu")

# Establecer todos los parámetros a None si alguna de las variables relevantes es None
if (torch_custom_fwd is None or torch_custom_bwd is None or
    torch_amp_custom_fwd is None or torch_amp_custom_bwd is None or
    custom_fwd is None or custom_bwd is None):
    
    torch_custom_fwd = None
    torch_custom_bwd = None
    torch_amp_custom_fwd = None
    torch_amp_custom_bwd = None
    custom_fwd = None
    custom_bwd = None

if (torch_custom_fwd is None or torch_custom_bwd is None) and (custom_fwd is None or custom_bwd is None):
    print("Advertencia: No se pudieron asignar correctamente 'torch_custom_fwd', 'torch_custom_bwd', 'custom_fwd' o 'custom_bwd'. Se omitirán los errores.")

print(f"torch_custom_fwd: {torch_custom_fwd}")
print(f"torch_custom_bwd: {torch_custom_bwd}")
print(f"torch_amp_custom_fwd: {torch_amp_custom_fwd}")
print(f"torch_amp_custom_bwd: {torch_amp_custom_bwd}")
print(f"custom_fwd: {custom_fwd}")
print(f"custom_bwd: {custom_bwd}")


import transformers.cache_utils
if hasattr(transformers.cache_utils, "DynamicCache") and \
    transformers.cache_utils.DynamicCache.__getitem__.__name__ != "__cache_utils_getitem__":

    source = inspect.getsource(transformers.cache_utils.DynamicCache.__getitem__)
    start = source.find("def")
    spaces = start*" "
    source = source.split("\n")
    source = "\n".join(x[start:] for x in source)
    where = source.find("raise KeyError")
    source = source[:where] + \
        f"if len(self) == 0:\n{spaces}{spaces}"\
        "    raise RuntimeError('Unsloth: You must call `FastLanguageModel.for_inference(model)` before doing inference for Unsloth models.')\n" + \
        f"{spaces}{spaces}else:\n{spaces}{spaces}{spaces}" + source[where:]
    source = source.replace("__getitem__", "__cache_utils_getitem__", 1)
    exec(source)
    transformers.cache_utils.DynamicCache.__getitem__ = __cache_utils_getitem__
pass

import bitsandbytes as bnb
from transformers import AutoTokenizer
from transformers.utils.import_utils import _is_package_available

# Detect if a GPU is available and its compute capability
try:
    if torch.cuda.is_available():
        major_version, minor_version = torch.cuda.get_device_capability()
        print(f"GPU detected with CUDA capability: {major_version}.{minor_version}")
    else:
        print("No GPU available. Using CPU.")
        major_version, minor_version = None, None 
except RuntimeError as e:
    print(f"Error checking device capability: {e}")
    major_version, minor_version = None, None 


SUPPORTS_BFLOAT16 = False
HAS_FLASH_ATTENTION = False
HAS_FLASH_ATTENTION_SOFTCAPPING = False

# Check for bfloat16 support and FlashAttention availability
if major_version is not None and major_version == 8:
    SUPPORTS_BFLOAT16 = True
    if _is_package_available("flash_attn"):
        try:
            from flash_attn.flash_attn_interface import flash_attn_cuda
            HAS_FLASH_ATTENTION = True

            from flash_attn import __version__ as flash_attn_version
            HAS_FLASH_ATTENTION_SOFTCAPPING = Version(flash_attn_version) >= Version("2.6.3")
            if not HAS_FLASH_ATTENTION_SOFTCAPPING:
                print(
                    "Unsloth: If you want to finetune Gemma 2, upgrade flash-attn to version 2.6.3 or higher!\n"\
                    "Newer versions support faster and less memory usage kernels for Gemma 2's attention softcapping!\n"\
                    "To update flash-attn, do the below:\n"\
                    '\npip install --no-deps --upgrade "flash-attn>=2.6.3"'
                )
        except:
            print(
                "Unsloth: Your Flash Attention 2 installation seems to be broken?\n"\
                "A possible explanation is you have a new CUDA version which isn't\n"\
                "yet compatible with FA2? Please file a ticket to Unsloth or FA2.\n"\
                "We shall now use Xformers instead, which does not have any performance hits!\n"\
                "We found this negligible impact by benchmarking on 1x A100."
            )

            import transformers.utils.import_utils
            transformers.utils.import_utils.is_flash_attn_2_available = lambda *args, **kwargs: False
            import transformers.utils
            transformers.utils.is_flash_attn_2_available = lambda *args, **kwargs: False

            HAS_FLASH_ATTENTION = False
        pass
    else:
        HAS_FLASH_ATTENTION = False
else:
    HAS_FLASH_ATTENTION = False
pass

from transformers.models.llama.modeling_llama import logger

from xformers import __version__ as xformers_version
if False: 
    raise ImportError(
        "Unsloth: If you are in Colab, we updated the top cell install instructions - please change it to below "\
        "then press Disconnect Runtime and then Restart it.\n"\
        "\n"\
        "%%capture\n"
        "# Installs Unsloth, Xformers (Flash Attention) and all other packages!\n"
        '!pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"\n'
        '!pip install --no-deps "xformers<=0.0.27" trl peft accelerate bitsandbytes\n'\
        '\n'\
        f"Otherwise in local machines, your xformers version of {xformers_version} is too new.\n"\
        'Please downgrade xformers via `pip install --force-reinstall "xformers<=0.0.27"'
    )
pass

# Check for xformers and torch version compatibility
if   Version(torch_version) < Version("2.2.0") and Version(xformers_version) >= Version("0.0.24"):
    raise ImportError(
        f"Unsloth: You have torch = {torch_version} but xformers = {xformers_version}.\n"\
        f"Please install xformers < 0.0.24 for torch = {torch_version}."
    )
elif Version(torch_version) < Version("2.3.0") and Version(xformers_version) >= Version("0.0.26"):
    raise ImportError(
        f"Unsloth: You have torch = {torch_version} but xformers = {xformers_version}.\n"\
        f"Please install xformers < 0.0.26 for torch = {torch_version}."
    )
elif Version(torch_version) < Version("2.4.0") and Version(xformers_version) > Version("0.0.27"):
    raise ImportError(
        f"Unsloth: You have torch = {torch_version} but xformers = {xformers_version}.\n"\
        f"Please install xformers <= 0.0.27 for torch = {torch_version}."
    )
pass

from xformers._cpp_lib import _register_extensions

import xformers.ops.fmha as xformers
xformers_attention = xformers.memory_efficient_attention

from trl import __version__ as trl_version
if False:
    raise ImportError(
        "Unsloth: If you are in Colab, we updated the top cell install instructions - please change it to below "\
        "then press Disconnect Runtime and then Restart it.\n"\
        "\n"\
        "%%capture\n"
        "# Installs Unsloth, Xformers (Flash Attention) and all other packages!\n"
        '!pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"\n'
        '!pip install --no-deps "xformers<=0.0.27" trl peft accelerate bitsandbytes\n'\
        '\n'\
        f"Otherwise in local machines, your TRL version of {trl_version} is too new.\n"\
        'Please downgrade TRL via `pip install --force-reinstall trl'
    )
pass

if Version(xformers_version) >= Version("0.0.27"):
    import accelerate.utils.operations
    if hasattr(accelerate.utils.operations, "send_to_device") and \
        accelerate.utils.operations.send_to_device.__name__ != "_fixed_send_to_device":
        from accelerate.utils.operations import *
        send_to_device = inspect.getsource(accelerate.utils.operations.send_to_device)
        send_to_device = re.sub(
            r"([ ]{4,})return tensor\.to\(device\)",
            r"\1try: return tensor.to(device)\n\1except: return tensor",
            send_to_device,
        ).replace("def send_to_device", "def _fixed_send_to_device")
        exec(send_to_device)
        accelerate.utils.operations.send_to_device = _fixed_send_to_device
    pass
pass

import functools
@functools.lru_cache(None)
def is_big_gpu(index):
    sms = torch.cuda.get_device_properties(index).multi_processor_count
    if sms < 80: 
        return False
    return True
import torch._inductor.utils
torch._inductor.utils.is_big_gpu = is_big_gpu

torch_compile_arguments = [
    "config.dce = True",
    "config.memory_planning = True",
    "config.memory_pool = 'combined'",
    "config.coordinate_descent_tuning = True",
    "config.max_autotune_gemm = False", 
    "config.autotune_multi_device = False",
    "config.max_autotune_gemm_backends = 'ATEN'", 
    "config.aggressive_fusion = False", 
    "config.cuda.enable_cuda_lto = True",
    "config.cuda.use_fast_math = True",
    "config.cuda.compile_opt_level = '-O2'",
]
torch_dynamo_arguments = [
    "config.accumulated_cache_size_limit = 512", 
    "config.suppress_errors = True", 
    "config.do_not_emit_runtime_asserts = True",
]
import torch._inductor.config as config
for _try_compile_argument in torch_compile_arguments:
    try:    exec(_try_compile_argument)
    except: pass
pass
import torch._dynamo.config as config
for _try_dynamo_argument in torch_dynamo_arguments:
    try:    exec(_try_dynamo_argument)
    except: pass
pass
torch_compile_options = {
    "epilogue_fusion"   : True,
    "max_autotune"      : True,
    "shape_padding"     : True,
    "trace.enabled"     : False, 
    "triton.cudagraphs" : False,
}

def prepare_model_for_kbit_training(
    model                      : Any,
    use_gradient_checkpointing : Optional = True,
    use_reentrant              : Optional[bool] = True,
) -> Any:
    

    with torch.no_grad():
        for name, param in model.named_parameters():
            if ".lora_A." in name or ".lora_B." in name or ".lora_magnitude_vector" in name:
                param.requires_grad_(True)
                if param.dtype != torch.float32:
                    name = name.replace("base_model", "model", 1)
                    layer_number = re.search(r"\.[\d]{1,}\.", name).group(0)
                    name = name.replace(layer_number, f"[{layer_number[1:-1]}].")
                    name = name.replace(".weight", "", 1)
                    exec(f"{name}.to(torch.float32)")
                pass
            else:
                param.requires_grad_(False)
        pass
    pass

    if use_gradient_checkpointing == "unsloth":

        original_model = model
        while hasattr(original_model, "model"):
            original_model._offloaded_gradient_checkpointing = True
            original_model = original_model.model
        pass
        original_model._offloaded_gradient_checkpointing = True
        
        model.gradient_checkpointing_enable()

    elif use_gradient_checkpointing == True:
        model.gradient_checkpointing_enable()
    pass

    if use_reentrant:
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:
            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)
            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

    return model
pass


def patch_tokenizer(model, tokenizer):
    
    possible_reserved_tokens = (
        "<|finetune_right_pad_id|>", 
        "<pad>",                     
        "<|reserved",                
        "<|placeholder",             
        "[control",                  
    )
    joiner = "\1\0=+=\0\1"
    number_repetitions = 3 - 1 

    if model is not None:
        model.config.update({"unsloth_version" : __version__})

    bad_pad_token = False
    if hasattr(tokenizer, "pad_token") and tokenizer.pad_token is not None:
        bad_pad_token = tokenizer.eos_token == tokenizer.pad_token
    elif hasattr(tokenizer, "pad_token") and tokenizer.pad_token is None:
        bad_pad_token = True
    else:
        bad_pad_token = False
    pass

    if bad_pad_token:
        added_tokens = [str(x) for x in tokenizer.added_tokens_decoder.values()]
        all_added_tokens = joiner.join(added_tokens[::-1])
        all_added_tokens += joiner

        final_pad_token  = None
        final_good_match = False

        for possible_reserved_token in possible_reserved_tokens:
            possible_reserved_token = re.escape(possible_reserved_token)
            found = re.finditer(f"{possible_reserved_token}", all_added_tokens)
            first_match = None
            good_match  = False
            for j, x in enumerate(found):
                if j == 0: first_match = x
                if j >= number_repetitions:
                    good_match = True
                    break
                pass
            pass

            if first_match is None: continue

            start = first_match.span(0)[0]
            possible_pad_token = first_match.group(0)
            end = all_added_tokens.find(joiner, start)
            first_match = all_added_tokens[start:end]

            if first_match is not None:
                good_match = possible_pad_token.endswith((">", "|>", "]", ")"))
            pass
            possible_pad_token = first_match

            if not final_good_match and good_match:
                final_good_match = True
                final_pad_token = possible_pad_token
                break
            else:
                final_good_match = False
                final_pad_token = possible_pad_token
            pass
        pass
        possible_pad_token = final_pad_token

        if possible_pad_token is None and hasattr(tokenizer, "unk_token"):
            possible_pad_token = tokenizer.unk_token
        pass

        if possible_pad_token is not None:
            check_pad_token = tokenizer(possible_pad_token, add_special_tokens = False).input_ids
            if len(check_pad_token) != 1:
                possible_pad_token = None
            if model is not None and check_pad_token[0] >= model.config.vocab_size:
                possible_pad_token = None
        pass

        if possible_pad_token is None:
            new_pad_token = "<|PAD_TOKEN|>"
            while new_pad_token in tokenizer.get_vocab():
                new_pad_token = f"<{new_pad_token}>"
            pass
            possible_pad_token = new_pad_token
        pass

        name = model.config._name_or_path if model is not None else "Model"
        logger.warning_once(
            f"{name} does not have a padding token! Will use pad_token = {possible_pad_token}."
        )
        
        tokenizer.add_special_tokens({"pad_token" : possible_pad_token})
        tokenizer.pad_token = possible_pad_token
        if model is not None:
            model.config.update({"pad_token_id" : tokenizer.pad_token_id})
            if getattr(model, "generation_config") is not None:
                model.generation_config.update(pad_token_id = tokenizer.pad_token_id)
    else:
        if model is not None:
            if model.config.pad_token_id is None:
                model.config.update({"pad_token_id" : tokenizer.pad_token_id})
                if getattr(model, "generation_config") is not None:
                    model.generation_config.update(pad_token_id = tokenizer.pad_token_id)
        pass
    pass

    if model is not None:
        if getattr(model, "generation_config") is not None:
            model.generation_config.update(max_length = model.config.max_position_embeddings)

    return model, tokenizer
pass

from peft import __version__ as peft_version
if Version(peft_version) < Version("0.12.0"):
    from peft.tuners.lora.layer import LoraLayer
    try:
        source = inspect.getsource(LoraLayer.update_layer)
        text = "if weight is not None:\n"
        start = source.find(text) + len(text)
        end = source.find("self.to(weight.device)", start)
        spaces = re.findall(r"^([ ]{1,})break", source, flags = re.MULTILINE)[0]
        source = source.replace(source[start : end], spaces)
        spaces = len(re.match(r"[\s]{1,}", source).group(0))
        lines = source.split("\n")
        source = "\n".join(x[spaces:] for x in lines)
        source = re.sub("([^\.])nn\.", r"\1torch.nn.", source)
        source = source.replace("def update_layer", "def LoraLayer_update_layer")
        exec(source, globals())

        from peft.tuners.lora.layer import LoraLayer
        LoraLayer.update_layer = LoraLayer_update_layer
        from peft.tuners.lora import LoraLayer
        LoraLayer.update_layer = LoraLayer_update_layer
    except:
        logger.warning_once(
            "Unsloth unsuccessfully patched LoraLayer.update_layer. Please file a bug report.\n"\
            "Luckily, your training run will still work in the meantime!"
        )
    pass
pass

import psutil
def _get_statistics(statistics = None, force_download = True):
    try:
        n_cpus = psutil.cpu_count(logical = False)

        keynames = "\n" + "\n".join(os.environ.keys())
        if statistics is not None: pass
        elif "\nCOLAB_"  in keynames and n_cpus == 1: statistics = "colab"
        elif "\nCOLAB_"  in keynames: statistics = "colabpro"
        elif "\nKAGGLE_" in keynames: statistics = "kaggle"
        elif "\nRUNPOD_" in keynames: statistics = "runpod"
        elif "\nAWS_"    in keynames: statistics = "aws"
        elif "\nAZURE_"  in keynames: statistics = "azure"
        elif "\nK_" in keynames or "\nFUNCTION_" in keynames: statistics = "gcp"
        elif "\nINVOCATION_ID" in keynames: statistics = "lambda"
        else: statistics = "other"

        if statistics is not None:
            from transformers import AutoModelForCausalLM
            stats_model = AutoModelForCausalLM.from_pretrained(
                f"unslothai/{statistics}",
                force_download = force_download,
            )
            del stats_model
        pass
    except:
        pass
pass


def get_statistics():
    from huggingface_hub.utils import disable_progress_bars, enable_progress_bars, are_progress_bars_disabled
    disabled = False
    if not are_progress_bars_disabled():
        disable_progress_bars()
        disabled = True
    pass
    _get_statistics(None)
    _get_statistics("repeat", force_download = False)
    try:
        vram = torch.cuda.get_device_properties(0).total_memory / 1024 / 1024 / 1024
        if   vram <= 8 : vram = 8
        elif vram <= 16: vram = 16
        elif vram <= 20: vram = 20
        elif vram <= 24: vram = 24
        elif vram <= 40: vram = 40
        elif vram <= 48: vram = 48
        elif vram <= 80: vram = 80
        else: vram = 96
        _get_statistics(f"vram-{vram}")
    except:
        pass
    pass
    try:
        devices = torch.cuda.device_count()
        _get_statistics(f"{devices if devices <= 8 else 9}")
    except:
        pass
    if disabled: enable_progress_bars()
pass


def _calculate_n_gradient_checkpoints(
    n_layers : int,
    method   : Optional[Union[str, int]] = "sqrt",
) -> List[int]:
    assert(type(n_layers) is int and n_layers > 0)

    if method is None: method = "sqrt"

    if method == "sqrt":
        n_checkpoints = int(n_layers**0.5)
    elif type(method) is int and method > 0:
        n_checkpoints = int(np.ceil(n_layers / method))
    else:
        raise ValueError("method must be 'sqrt' or an int >0 and <= n_layers.")

    size = n_layers // n_checkpoints
    sizes = np.full(n_checkpoints, size, dtype = int)
    leftovers = n_layers % n_checkpoints
    for k in range(leftovers):
        sizes[n_checkpoints-1-k] += 1
    boundaries = np.hstack((0, np.cumsum(sizes)))
    boundaries = boundaries.tolist()
    return boundaries
pass


def calculate_n_gradient_checkpoints(
    n_layers              : int,
    layers_per_checkpoint : Optional[Union[str, int]] = "sqrt",
) -> List[int]:
    assert(type(n_layers) is int and n_layers > 0)

    if layers_per_checkpoint is None or layers_per_checkpoint == 1:
        return None

    boundaries = _calculate_n_gradient_checkpoints(n_layers, layers_per_checkpoint)

    assert(boundaries[0] == 0 and boundaries[-1] == n_layers)
    assert(min(boundaries) == 0 and max(boundaries) == n_layers)
    assert(np.diff(boundaries).min() >= 0)
    return boundaries
pass


def prepare_n_gradient_checkpoints(
    model                 : Any,
    layers_per_checkpoint : Optional[Union[str, int]] = "sqrt",
    use_reentrant         : Optional[bool] = True,
) -> None:
    _model = None
    if hasattr(model, "layers"):
        _model = model
    elif hasattr(model, "model"):
        if hasattr(model.model, "layers"):
            _model = model.model
    if _model is None:
        raise TypeError("`model` or `model.model` does not have attribute `layers`. Are you sure this is a model?")
    pass

    if use_reentrant is False:
        use_reentrant = True
    pass

    n_layers = len(_model.layers)
    boundaries = calculate_n_gradient_checkpoints(n_layers, layers_per_checkpoint)
    _model._gradient_checkpointing_boundaries    = boundaries
    _model._gradient_checkpointing_use_reentrant = use_reentrant
pass


class Unsloth_Offloaded_Gradient_Checkpointer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, forward_function, hidden_states, *args):
        saved_hidden_states = hidden_states.to("cpu", non_blocking = True)
        with torch.no_grad():
            output = forward_function(hidden_states, *args)
        ctx.save_for_backward(saved_hidden_states)
        ctx.forward_function = forward_function
        ctx.args = args
        return output
    pass

    @staticmethod
    def backward(ctx, dY):
        (hidden_states,) = ctx.saved_tensors
        hidden_states = hidden_states.to("cpu", non_blocking = True).detach()
        hidden_states.requires_grad = True
        with torch.enable_grad():
            (output,) = ctx.forward_function(hidden_states, *ctx.args)
        torch.autograd.backward(output, dY)
        return (None, hidden_states.grad,) + (None,)*len(ctx.args)
    pass
pass


@torch._disable_dynamo
def unsloth_offloaded_gradient_checkpoint(function, *args, use_reentrant = None, **kwargs):
    return Unsloth_Offloaded_Gradient_Checkpointer.apply(function, *args)
pass

from transformers.utils.quantization_config import BitsAndBytesConfig, QuantizationMethod
from inspect import getsource
from accelerate.utils.dataclasses import DistributedType
BitsAndBytesConfig__init__ = getsource(BitsAndBytesConfig.__init__)
BitsAndBytesConfig__init__ = re.sub(
    r"if[\s]{1,}kwargs\:[\s]{1,}.+?\n",
    "",
    BitsAndBytesConfig__init__,
    flags = re.MULTILINE,
)
BitsAndBytesConfig__init__ = BitsAndBytesConfig__init__.split("\n")
length_spaces = len(re.match(r"[\s]{1,}", BitsAndBytesConfig__init__[0]).group(0))
BitsAndBytesConfig__init__ = "\n".join(x[length_spaces:] for x in BitsAndBytesConfig__init__)
BitsAndBytesConfig__init__ = BitsAndBytesConfig__init__.replace(
    "__init__",
    "_BitsAndBytesConfig__init__",
)

def _prepare_backend(
    self, cpu: bool = False, sagemaker_dp = False, backend: str = None,
) -> tuple[str, DistributedType]:
    return None, DistributedType.NO
pass
import accelerate.state
accelerate.state.PartialState._prepare_backend = _prepare_backend

import accelerate.accelerator
prepare = inspect.getsource(accelerate.accelerator.Accelerator.prepare)
prepare = prepare.split("\n")
spaces = prepare[0].find("def")
prepare = "\n".join(x[spaces:] for x in prepare)
x = "for obj in args:"
s = " "*spaces
prepare = prepare.replace(x, f'self.state.distributed_type = DistributedType.NO\n{s}{x}', 1)
exec(prepare, globals())
accelerate.accelerator.Accelerator.prepare = prepare

exec(BitsAndBytesConfig__init__, globals())

import transformers.utils.quantization_config
transformers.utils.quantization_config.BitsAndBytesConfig.__init__ = _BitsAndBytesConfig__init__

import pickle

def offload_to_disk(W, model, name, temporary_location : str = "_unsloth_temporary_saved_buffers"):
    file_location = os.path.join(temporary_location, model.config._name_or_path)
    if not os.path.exists(file_location):
        os.makedirs(file_location)
    pass

    filename = os.path.join(file_location, f"{name}.pt")
    W = W.weight if hasattr(W, "weight") else W
    torch.save(W, filename, pickle_module = pickle, pickle_protocol = pickle.HIGHEST_PROTOCOL,)
    offloaded_W = torch.load(filename, map_location = "cpu", mmap = True)
    offloaded_W._offloaded_file_location = filename
    return offloaded_W
pass


def offload_input_embeddings(model, temporary_location : str = "_unsloth_temporary_saved_buffers"):
    offloaded_W = offload_to_disk(model.get_input_embeddings(), model, "input_embeddings", temporary_location)
    new_input_embeddings = torch.nn.Embedding.from_pretrained(offloaded_W)
    new_input_embeddings._offloaded_file_location = offloaded_W._offloaded_file_location
    model.set_input_embeddings(new_input_embeddings)
    return
pass


def offload_output_embeddings(model, temporary_location : str = "_unsloth_temporary_saved_buffers"):
    offloaded_W = offload_to_disk(model.get_output_embeddings(), model, "output_embeddings", temporary_location)

    new_output_embeddings = torch.nn.Linear(1, 1, bias = None)
    del new_output_embeddings.weight
    new_output_embeddings.weight = offloaded_W
    new_output_embeddings.in_features  = offloaded_W.shape[1]
    new_output_embeddings.out_features = offloaded_W.shape[0]

    new_output_embeddings._offloaded_file_location = offloaded_W._offloaded_file_location
    model.set_output_embeddings(new_output_embeddings)
    return
pass

def is_bfloat16_supported():
    return SUPPORTS_BFLOAT16
pass

def patch_linear_scaling(
    model_name = "gemma2",
    rope_module = None,
    scaled_rope_module = None,
    attention_module = None,
):
    assert(rope_module is not None and scaled_rope_module is not None)
    assert(attention_module is not None)

    rope_name = rope_module.__name__
    scaled_rope_name = scaled_rope_module.__name__
    model_filepath = f"transformers.models.{model_name}.modeling_{model_name}"
    exec_code = \
        f"import torch.nn as nn\n"\
        f"from typing import Union, Optional, List, Any, Callable, Tuple\n"\
        f"from {model_filepath} import logger, "\
        f"{model_name.title()}Attention, {model_name.title()}Config"

    try:
        function = inspect.getsource(attention_module.__init__)
    except:
        return None, None
    where = function.find("def")
    function = function.split("\n")
    function = "\n".join(x[where:] for x in function)
    init_name = f"{model_name.title()}Attention__init__"
    function = function.replace("def __init__", f"def {init_name}")
    function = function.replace(
        "super().__init__()",
        f"super({model_name.title()}Attention, self).__init__()",
    )
    fix_rope_function = """
    if getattr(self.config, "rope_scaling", None) is None:
        self.rotary_emb = {rope_function}(
            dim = self.head_dim,
            max_position_embeddings=self.max_position_embeddings,
            base=self.rope_theta,
        )
    else:
        scaling_type = self.config.rope_scaling["type"]
        scaling_factor = self.config.rope_scaling["factor"]
        if scaling_type == "linear":
            self.rotary_emb = {scaled_rope_function}(
                dim = self.head_dim,
                max_position_embeddings=self.max_position_embeddings,
                scaling_factor=scaling_factor,
                base=self.rope_theta,
            )
        else:
            raise ValueError(f"Unknown RoPE scaling type {{scaling_type}}")
    pass
    """
    fix_rope_function = fix_rope_function.format(
        rope_function        = rope_module.__name__,
        scaled_rope_function = scaled_rope_module.__name__,
    )
    rotary_emb = re.findall(
        "self.rotary_emb = .+?\)", function,
        flags = re.DOTALL | re.MULTILINE,
    )
    if len(rotary_emb) == 0: return None, function
    rotary_emb = rotary_emb[0]
    function = function.replace(rotary_emb, fix_rope_function, 1)
    function = exec_code + "\n\n" + function
    return init_name, function
pass

def patch_llama_rope_scaling(
    model_name = "llama",
    rope_module = None,
    scaled_rope_module = None,
    extended_rope_module = None,
    attention_module = None,
):
    assert(\
        rope_module is not None and \
        scaled_rope_module is not None and \
        extended_rope_module is not None
    )
    assert(attention_module is not None)

    rope_name = rope_module.__name__
    scaled_rope_name = scaled_rope_module.__name__
    model_filepath = f"transformers.models.{model_name}.modeling_{model_name}"
    exec_code = \
        f"import torch.nn as nn\n"\
        f"from typing import Union, Optional, List, Any, Callable, Tuple\n"\
        f"from {model_filepath} import logger, "\
        f"{model_name.title()}Attention, {model_name.title()}Config"

    try:
        function = inspect.getsource(attention_module.__init__)
    except:
        return None, None
    where = function.find("def")
    function = function.split("\n")
    function = "\n".join(x[where:] for x in function)
    init_name = f"{model_name.title()}Attention__init__"
    function = function.replace("def __init__", f"def {init_name}")
    function = function.replace(
        "super().__init__()",
        f"super({model_name.title()}Attention, self).__init__()",
    )
    fix_rope_function = """
    if getattr(self.config, "rope_scaling", None) is None:
        self.rotary_emb = {rope_function}(
            dim = self.head_dim,
            max_position_embeddings=self.max_position_embeddings,
            base=self.rope_theta,
        )
    else:
        scaling_type1 = self.config.rope_scaling.get("type", None)
        scaling_type2 = self.config.rope_scaling.get("rope_type", None)
        scaling_type = scaling_type1 if scaling_type1 is not None else scaling_type2
        scaling_factor = self.config.rope_scaling.get("factor")

        if scaling_type == "linear":
            self.rotary_emb = {scaled_rope_function}(
                dim = self.head_dim,
                max_position_embeddings=self.max_position_embeddings,
                scaling_factor=scaling_factor,
                base=self.rope_theta,
            )
        elif scaling_type == "llama3":
            self.rotary_emb = {extended_rope_function}(
                dim = self.head_dim,
                max_position_embeddings=self.max_position_embeddings,
                base=self.rope_theta,
            )
        else:
            raise ValueError(f"Unknown RoPE scaling type {{scaling_type}}")
    pass
    """
    fix_rope_function = fix_rope_function.format(
        rope_function          = rope_module.__name__,
        scaled_rope_function   = scaled_rope_module.__name__,
        extended_rope_function = extended_rope_module.__name__,
    )
    rotary_emb = re.findall(
        "self.rotary_emb = .+?\)", function,
        flags = re.DOTALL | re.MULTILINE,
    )
    if len(rotary_emb) == 0: return None, function
    rotary_emb = rotary_emb[0]
    function = function.replace(rotary_emb, fix_rope_function, 1)
    function = exec_code + "\n\n" + function
    return init_name, function
pass


def check_gpu_memory():  # Renamed function
    output = np.array([0,])
    try:
        output = subprocess.check_output("nvidia-smi --query-gpu=memory.used --format=csv", shell = True)
        output = re.findall(rb'([\d]{1,})[\s]{1,}MiB', output)
        output = np.array([int(x.decode('utf-8'))/1024 for x in output])
    except:
        # If nvidia-smi is not available, assume no GPU is being used
        pass  
    return output
pass
PRE_CHECK = check_gpu_memory()


def create_boolean_mask(n = 4096, sliding_window = 2048):
    mask = torch.ones(n, n, dtype = torch.bool)
    if sliding_window == 0:
        return torch.triu(mask, diagonal = 1, out = mask)
    pass
    torch.triu(mask, diagonal = 0, out = mask)
    torch.triu(mask.T, diagonal = -sliding_window, out = mask.T)
    mask = mask.T
    torch.logical_not(mask, out = mask)
    return mask
pass


def test_mask_creation():
    from transformers.modeling_attn_mask_utils import AttentionMaskConverter
    for n in range(2, 23):
        for s in range(1, 23):
            correct_mask = AttentionMaskConverter(
                is_causal = True,
                sliding_window = s,
            ).to_causal_4d(1, n, n, dtype = torch.float16,).squeeze(0).squeeze(0)
            correct_mask = (correct_mask == correct_mask.min())
            our_mask = create_boolean_mask(n = n, sliding_window = s)
            assert(torch.all(correct_mask == our_mask))
        pass
        correct_mask = AttentionMaskConverter(
            is_causal = True,
            sliding_window = None,
        ).to_causal_4d(1, n, n, dtype = torch.float16,).squeeze(0).squeeze(0)
        correct_mask = (correct_mask == correct_mask.min())
        our_mask = create_boolean_mask(n = n, sliding_window = 0)
        assert(torch.all(correct_mask == our_mask))
    pass
pass
