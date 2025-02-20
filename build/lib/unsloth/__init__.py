import warnings, importlib, sys
from packaging.version import Version
import os, re, subprocess, inspect
import numpy as np

# Verificación de compatibilidad con PyTorch
try:
    import torch
except ImportError:
    raise ImportError("Pytorch no está instalado. Visita https://pytorch.org/.\n"\
                      "Tenemos algunas instrucciones de instalación en nuestra página de Github.")
pass

# Verificación de la versión de PyTorch
torch_version = torch.__version__.split(".")
major_torch, minor_torch = torch_version[0], torch_version[1]
major_torch, minor_torch = int(major_torch), int(minor_torch)
if major_torch < 2:
    raise ImportError("Unsloth solo soporta PyTorch 2 o superior. Actualiza tu PyTorch a 2.1.\n"\
                      "Tenemos algunas instrucciones de instalación en nuestra página de Github.")
pass

# Verificación del soporte para bf16 (precisión de punto flotante)
SUPPORTS_BFLOAT16 = False

# Ajuste para la verificación del soporte bf16 en CUDA
old_is_bf16_supported = torch.cuda.is_bf16_supported if hasattr(torch.cuda, "is_bf16_supported") else lambda: False
if "including_emulation" in str(inspect.signature(old_is_bf16_supported)):
    def is_bf16_supported(including_emulation=False):
        return old_is_bf16_supported(including_emulation)
    torch.cuda.is_bf16_supported = is_bf16_supported
else:
    def is_bf16_supported(): return SUPPORTS_BFLOAT16
    torch.cuda.is_bf16_supported = is_bf16_supported
pass

# Importando bitsandbytes, que depende de CUDA para operaciones en baja precisión (4-bit y 8-bit)
import bitsandbytes as bnb

# Verificación de soporte de operaciones CUDA específicas para bitsandbytes
try:
    cdequantize_blockwise_fp32 = bnb.functional.lib.cdequantize_blockwise_fp32
except:
    warnings.warn(
        "Unsloth: No se pudo vincular correctamente CUDA, pero continuaremos sin soporte CUDA."
    )
pass

# Importando otros módulos necesarios para el funcionamiento del modelo
from .models import *
from .save import *
from .chat_templates import *
from .tokenizer_utils import *
from .trainer import *
