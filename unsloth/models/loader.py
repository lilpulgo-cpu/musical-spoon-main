from transformers import AutoConfig, AutoModel, AutoTokenizer
from transformers import __version__ as transformers_version
from peft import PeftConfig, PeftModel
from packaging.version import Version
import os
import torch

transformers_version = Version(transformers_version)

def get_model_name(model_name, load_in_4bit=True):
    # Si no hay GPU, no cargar en 4-bit
    if not torch.cuda.is_available() and load_in_4bit:
        print("No GPU found. Switching to 8-bit model.")
        load_in_4bit = False

    return model_name

class FastLanguageModel:
    @staticmethod
    def from_pretrained(
        model_name="unsloth/llama-3-8b-bnb-4bit",
        max_seq_length=None,
        dtype=None,
        load_in_4bit=True,
        token=None,
        device_map="sequential",
        rope_scaling=None,
        fix_tokenizer=True,
        trust_remote_code=False,
        use_gradient_checkpointing="unsloth",
        resize_model_vocab=None,
        revision=None,
        *args, **kwargs,
    ):
        if token is None and "HF_TOKEN" in os.environ:
            token = os.environ["HF_TOKEN"]

        if token is None and "HUGGINGFACE_TOKEN" in os.environ:
            token = os.environ["HUGGINGFACE_TOKEN"]

        model_name = get_model_name(model_name, load_in_4bit)

        try:
            model_config = AutoConfig.from_pretrained(
                model_name,
                token=token,
                revision=revision,
                trust_remote_code=trust_remote_code,
            )
            is_model = True
        except Exception as e:
            print(f"Error loading model config: {e}")
            model_config = None
            is_model = False

        try:
            peft_config = PeftConfig.from_pretrained(
                model_name,
                token=token,
                revision=revision,
                trust_remote_code=trust_remote_code,
            )
            is_peft = True
        except Exception as e:
            print(f"Error loading PEFT config: {e}")
            peft_config = None
            is_peft = False

        tokenizer_name = model_name if os.path.exists(os.path.join(model_name, "tokenizer_config.json")) else model_name

        try:
            model = AutoModel.from_pretrained(
                model_name,
                config=model_config,
                revision=revision,
                trust_remote_code=trust_remote_code,
                *args, **kwargs,
            )
        except Exception as e:
            print(f"Error loading model: {e}")
            model = None

        try:
            tokenizer = AutoTokenizer.from_pretrained(
                tokenizer_name,
                use_fast=True,
                revision=revision,
                trust_remote_code=trust_remote_code,
            )
        except Exception as e:
            print(f"Error loading tokenizer: {e}")
            tokenizer = None

        if resize_model_vocab is not None and model is not None:
            try:
                model.resize_token_embeddings(resize_model_vocab)
            except Exception as e:
                print(f"Error resizing token embeddings: {e}")

        if is_peft and model is not None:
            try:
                model = PeftModel.from_pretrained(
                    model,
                    model_name,
                    token=token,
                    revision=revision,
                    is_trainable=True,
                    trust_remote_code=trust_remote_code,
                )
                if use_gradient_checkpointing:
                    model.gradient_checkpointing_enable()
            except Exception as e:
                print(f"Error loading PEFT model: {e}")

        return model, tokenizer

    @staticmethod
    def get_peft_model(model_name, token=None, revision=None, **kwargs):
        """
        Método para obtener un modelo PEFT desde el nombre del modelo.
        """
        if not model_name:
            raise ValueError("Model name must be provided.")
        
        if token is None and "HF_TOKEN" in os.environ:
            token = os.environ["HF_TOKEN"]

        if token is None and "HUGGINGFACE_TOKEN" in os.environ:
            token = os.environ["HUGGINGFACE_TOKEN"]

        if not token:
            raise ValueError("Token must be provided or set in environment variables.")
        
        try:
            # Validar los argumentos adicionales para evitar errores
            if kwargs:
                print(f"Warning: Unrecognized arguments passed to get_peft_model: {kwargs}")

            # Cargar la configuración PEFT
            peft_config = PeftConfig.from_pretrained(
                model_name,
                token=token,
                revision=revision
            )
            
            # Cargar el modelo base
            model = AutoModel.from_pretrained(
                model_name,
                config=peft_config,
                revision=revision
            )
            
            # Cargar el modelo PEFT
            peft_model = PeftModel.from_pretrained(
                model,
                model_name,
                token=token,
                revision=revision
            )
            return peft_model
        
        except ValueError as e:
            print(f"ValueError in get_peft_model: {e}")
        except TypeError as e:
            print(f"TypeError in get_peft_model: {e}")
        except Exception as e:
            print(f"Error loading PEFT model: {e}")

        return None
