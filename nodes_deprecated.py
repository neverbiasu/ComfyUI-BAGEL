import os
import sys
import torch
import numpy as np
import random
import subprocess
import importlib
from typing import Dict, Tuple, Optional, Any
from PIL import Image
from folder_paths import folder_names_and_paths, models_dir as comfy_models_dir
from comfy.utils import ProgressBar
from accelerate import (
    infer_auto_device_map,
    load_checkpoint_and_dispatch,
    init_empty_weights,
    dispatch_model,
)
from accelerate.utils import BnbQuantizationConfig, load_and_quantize_model

# Add current directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

# Import BAGEL related modules
from data.data_utils import add_special_tokens, pil_img2rgb
from data.transforms import ImageTransform
from inferencer import InterleaveInferencer
from modeling.autoencoder import load_ae
from modeling.bagel import (
    BagelConfig,
    Bagel,
    Qwen2Config,
    Qwen2ForCausalLM,
    SiglipVisionConfig,
    SiglipVisionModel,
)
from modeling.qwen2 import Qwen2Tokenizer

def _register_bagel_model_folder() -> None:
    """Register BAGEL model paths without clobbering extra_model_paths.yaml."""

    default_path = os.path.join(comfy_models_dir, "bagel")
    paths, extensions = folder_names_and_paths.get("bagel", ([], []))
    merged_paths = list(paths)
    if default_path not in merged_paths:
        merged_paths.append(default_path)

    merged_extensions = list(extensions)
    for extension in (".json", ".safetensors"):
        if extension not in merged_extensions:
            merged_extensions.append(extension)

    folder_names_and_paths["bagel"] = (merged_paths, merged_extensions)


_register_bagel_model_folder()


_DEPRECATION_NOTICE_EMITTED = False


def _warn_deprecated_nodes_once() -> None:
    """Make the legacy path visible without disrupting old workflow loading."""
    global _DEPRECATION_NOTICE_EMITTED
    if not _DEPRECATION_NOTICE_EMITTED:
        print(
            "[BAGEL] Deprecated all-in-one nodes are loaded for legacy workflows. "
            "Use the native BAGEL nodes with official VAEEncode/VAEDecode for new graphs."
        )
        _DEPRECATION_NOTICE_EMITTED = True


_warn_deprecated_nodes_once()


def _require_dfloat11():
    """Lazily import ``DFloat11Model`` only when a legacy DF11 model is requested.

    Importing ``dfloat11`` pulls in ``cupy`` (CUDA-only), so it must never happen
    at custom-node import time -- the native BF16 path and the importable package
    must work without CUDA-only dependencies installed.
    """
    try:
        from dfloat11 import DFloat11Model
    except ImportError as exc:
        raise ImportError(
            "The selected model requires DFloat11, but the 'dfloat11' package is "
            "not installed (it needs CUDA + cupy). Install it on a CUDA host with "
            "'pip install dfloat11 --no-build-isolation', or pick a standard BF16 "
            "BAGEL model instead."
        ) from exc
    return DFloat11Model


def discover_bagel_model_dirs() -> Dict[str, str]:
    """Discover local BAGEL model directories under the configured models/bagel folder.

    Returns a mapping from folder-name -> absolute-path.
    """
    base_repo_dir = os.path.join(comfy_models_dir, "bagel")
    discovered: Dict[str, str] = {}
    try:
        if os.path.exists(base_repo_dir):
            for name in sorted(os.listdir(base_repo_dir)):
                p = os.path.join(base_repo_dir, name)
                if os.path.isdir(p):
                    discovered[name] = p
    except Exception as e:
        print(f"Error discovering bagel model dirs: {e}")
    return discovered


def is_df11_name(name: str) -> bool:
    """Heuristic to detect DFloat11 derived model names.

    Matches folder or repo names containing dfloat11
    """
    if not name:
        return False
    low = name.lower()
    kws = ["dfloat11", "df11", "df11-", "df11_"]
    return any(k in low for k in kws)


def is_echo4o_name(name: str) -> bool:
    """Heuristic to detect Echo-4o model names.

    Matches folder or repo names containing echo-4o, echo4o
    """
    if not name:
        return False
    low = name.lower()
    kws = ["echo-4o", "echo4o", "yejy53/echo-4o"]
    return any(k in low for k in kws)


def set_seed(seed: int) -> int:
    """Set random seeds for reproducibility"""
    if seed > 0:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    return seed


def pil_to_tensor(img: Image.Image) -> torch.Tensor:
    """Convert PIL image to ComfyUI tensor format"""
    img_array = np.array(img).astype(np.float32) / 255.0
    if len(img_array.shape) == 3:
        img_tensor = torch.from_numpy(img_array)[None,]  # Add batch dimension
    else:
        img_tensor = torch.from_numpy(img_array)
    return img_tensor


def tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
    """Convert ComfyUI tensor to PIL image"""
    if len(tensor.shape) == 4:
        tensor = tensor[0]  # Remove batch dimension
    img_array = (tensor.cpu().numpy() * 255).astype(np.uint8)
    return Image.fromarray(img_array)


def get_image_shapes_map() -> Dict[str, Tuple[int, int]]:
    """Get mapping of aspect ratios to image dimensions"""
    return {
        "1:1": (1024, 1024),
        "4:3": (768, 1024),
        "3:4": (1024, 768),
        "16:9": (576, 1024),
        "9:16": (1024, 576),
    }


def find_vae_path(model_dir: str) -> Optional[str]:
    """Find VAE model file in standard locations"""
    common_vae_dir = os.path.join(comfy_models_dir, "vae")
    common_vae_file = os.path.join(common_vae_dir, "ae.safetensors")

    potential_vae_paths = [
        os.path.join(model_dir, "vae", "ae.safetensors"),
        os.path.join(model_dir, "ae.safetensors"),
        common_vae_file,
    ]

    for vae_path in potential_vae_paths:
        if os.path.exists(vae_path):
            return vae_path

    return None


def load_vae_model(model_dir: str):
    """Load VAE model from standard locations"""
    vae_path = find_vae_path(model_dir)
    if not vae_path:
        potential_paths = [
            os.path.join(model_dir, "vae", "ae.safetensors"),
            os.path.join(model_dir, "ae.safetensors"),
            os.path.join(comfy_models_dir, "vae", "ae.safetensors"),
        ]
        raise FileNotFoundError(
            f"VAE model (ae.safetensors) could not be found in any of the expected paths: {potential_paths}"
        )

    try:
        vae_model, vae_config = load_ae(local_path=vae_path)
        if vae_model is None or vae_config is None:
            raise ValueError(f"Failed to load VAE from {vae_path}")
        return vae_model, vae_config, vae_path
    except Exception as e:
        raise RuntimeError(f"Error loading VAE from {vae_path}: {e}")


def create_error_response_image() -> torch.Tensor:
    """Create a default error response image tensor"""
    return torch.zeros((1, 512, 512, 3))


def validate_seed(seed: int) -> bool:
    """Validate seed parameter"""
    return isinstance(seed, int) and seed >= 0


def validate_cfg_scale(scale: float, min_val: float = 1.0, max_val: float = 8.0) -> bool:
    """Validate CFG scale parameter"""
    return isinstance(scale, (int, float)) and min_val <= scale <= max_val


def validate_timesteps(timesteps: int, min_val: int = 10, max_val: int = 100) -> bool:
    """Validate timesteps parameter"""
    return isinstance(timesteps, int) and min_val <= timesteps <= max_val


def validate_temperature(temp: float) -> bool:
    """Validate temperature parameter"""
    return isinstance(temp, (int, float)) and 0.0 <= temp <= 1.0


def validate_max_tokens(tokens: int, min_val: int = 64, max_val: int = 4096) -> bool:
    """Validate max tokens parameter"""
    return isinstance(tokens, int) and min_val <= tokens <= max_val


def setup_inference_progress_bar(num_timesteps: int) -> ProgressBar:
    """Setup progress bar for inference operations"""
    actual_iterations = num_timesteps - 1 if num_timesteps > 0 else 0
    return ProgressBar(actual_iterations)


def create_common_inference_config(
    cfg_text_scale: float,
    num_timesteps: int,
    cfg_interval: float = 0.4,
    timestep_shift: float = 3.0,
    cfg_renorm_min: float = 0.0,
    cfg_renorm_type: str = "global",
    text_temperature: float = 0.3,
    show_thinking: bool = False,
    **kwargs
) -> Dict[str, Any]:
    """Create common inference configuration dictionary"""
    config = {
        "max_think_token_n": 1024 if show_thinking else 1024,
        "do_sample": False if not show_thinking else False,
        "text_temperature": text_temperature if show_thinking else 0.3,
        "cfg_text_scale": cfg_text_scale,
        "cfg_interval": [cfg_interval, 1.0],  # End value fixed at 1.0
        "timestep_shift": timestep_shift,
        "num_timesteps": num_timesteps,
        "cfg_renorm_min": cfg_renorm_min,
        "cfg_renorm_type": cfg_renorm_type,
    }

    # Add additional parameters
    config.update(kwargs)

    return config


def download_model_with_git(
    model_dir: str, repo_id: str = "ByteDance-Seed/BAGEL-7B-MoT"
) -> str:
    """
    Download model using git lfs (recommended method)

    Args:
        model_dir: Directory to download the repo to (repo files will be placed directly here)
        repo_id: Hugging Face repository ID

    Returns:
        Path to the downloaded model if successful, None otherwise
    """
    try:
        print(f"Downloading BAGEL model using git lfs to {model_dir}...")

        # Create parent directory if it doesn't exist
        os.makedirs(model_dir, exist_ok=True)

        # Check if git lfs is installed
        try:
            subprocess.run(["git", "lfs", "version"], check=True, capture_output=True)
        except (subprocess.CalledProcessError, FileNotFoundError):
            print("Git LFS not found. Installing git lfs...")
            subprocess.run(["git", "lfs", "install"], check=True)

        # Clone the repository directly to model_dir
        clone_cmd = ["git", "clone", f"https://huggingface.co/{repo_id}", model_dir]

        result = subprocess.run(clone_cmd, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"Successfully downloaded BAGEL model to {model_dir}")
            return model_dir
        else:
            print(f"Git clone failed: {result.stderr}")
            return None

    except Exception as e:
        print(f"Error downloading model with git: {e}")
        return None


def fix_nested_model_structure(model_dir: str, repo_id: str) -> bool:
    """
    Fix nested model directory structure if exists.

    Some downloads may create nested folders like models/bagel/Echo-4o/Echo-4o/
    This function detects and fixes such structure to models/bagel/Echo-4o/

    Args:
        model_dir: The expected model directory path
        repo_id: Repository ID (e.g., "Yejy53/Echo-4o")

    Returns:
        True if structure was fixed or already correct, False if error
    """
    if not os.path.exists(model_dir):
        return False

    try:
        repo_name = repo_id.split("/")[-1]
        nested_path = os.path.join(model_dir, repo_name)

        # Check if nested structure exists
        if os.path.exists(nested_path) and os.path.isdir(nested_path):
            nested_files = os.listdir(nested_path)
            main_files = [f for f in os.listdir(model_dir) if f != repo_name and not f.startswith('.')]

            # If main directory only contains the nested folder and some metadata
            if len(main_files) == 0 and nested_files:
                print(f"Fixing nested directory structure: {nested_path} -> {model_dir}")

                # Move all files from nested directory to parent
                import shutil
                for item in nested_files:
                    src = os.path.join(nested_path, item)
                    dst = os.path.join(model_dir, item)
                    if os.path.exists(dst):
                        if os.path.isdir(dst):
                            shutil.rmtree(dst)
                        else:
                            os.remove(dst)
                    shutil.move(src, dst)

                # Remove empty nested directory
                os.rmdir(nested_path)
                print("Successfully fixed nested directory structure")
                return True

        return True

    except Exception as e:
        print(f"Error fixing nested structure: {e}")
        return False


def download_model_with_hf_hub(
    model_dir: str, repo_id: str = "ByteDance-Seed/BAGEL-7B-MoT"
) -> str:
    """
    Download model using huggingface_hub with resume support

    Args:
        model_dir: Directory to download the repo to (repo files will be placed directly here)
        repo_id: Hugging Face repository ID

    Returns:
        Path to the downloaded model if successful, None otherwise
    """
    try:
        from huggingface_hub import snapshot_download

        print(f"Downloading model from {repo_id} using huggingface_hub to {model_dir}...")

        # Create parent directory if it doesn't exist
        os.makedirs(model_dir, exist_ok=True)

        # Download with resume support and better error handling
        downloaded_path = snapshot_download(
            repo_id=repo_id,
            local_dir=model_dir,
            local_dir_use_symlinks=False,
            resume_download=True,  # Enable resume functionality
            force_download=False,  # Don't re-download existing files
        )

        # Fix any nested directory structure
        fix_nested_model_structure(model_dir, repo_id)

        print(f"Successfully downloaded model from {repo_id} to {model_dir}")
        return model_dir

    except ImportError:
        print(
            "huggingface_hub not installed. Please install it with: pip install huggingface_hub"
        )
        return None
    except Exception as e:
        print(f"Error downloading model {repo_id} with huggingface_hub: {e}")
        return None


def check_model_files(model_path: str, is_df11_model: bool) -> bool:
    """
    Check if core model configuration files exist.
    VAE and main weights (ema.safetensors for standard) are checked separately during load.

    Args:
        model_path: Path to the model directory
        is_df11_model: Boolean indicating if the model is DFloat11

    Returns:
        True if core config files exist, False otherwise
    """
    required_files = [
        "llm_config.json",
        "vit_config.json",
    ]

    # DFloat11 models do not have ema.safetensors in their root.
    # Standard models expect ema.safetensors.
    # VAE presence is checked more robustly during the loading process itself.
    if not is_df11_model:
        required_files.append("ema.safetensors")

    for file_name in required_files:
        if not os.path.exists(os.path.join(model_path, file_name)):
            print(f"Missing required model file: {os.path.join(model_path, file_name)}")
            return False

    return True


def calculate_optimal_memory_gb(quantization_mode: str) -> float:
    """
    Calculate optimal memory usage based on available GPU memory and quantization mode.

    Args:
        quantization_mode: Quantization mode (1=BF16, 2=NF4, 3=INT8, 4=FP8)

    Returns:
        Optimal memory usage in GB per GPU
    """
    try:
        if not torch.cuda.is_available():
            return 8.0  # CPU fallback

        # Get total GPU memory
        total_memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)

        # Reserve memory for system and other processes
        if quantization_mode == "BF16":  # BF16
            # Full precision needs more headroom
            reserved_ratio = 0.15  # Reserve 15%
            optimal_ratio = 0.8  # Use 80% of remaining
        elif quantization_mode == "NF4":  # NF4
            # 4-bit quantization is very memory efficient
            reserved_ratio = 0.1  # Reserve 10%
            optimal_ratio = 0.85  # Use 85% of remaining
        elif quantization_mode == "INT8" or quantization_mode == "FP8":  # INT8
            # 8-bit quantization is moderately efficient
            reserved_ratio = 0.1  # Reserve 10%
            optimal_ratio = 0.82  # Use 82% of remaining
        else:
            # Default conservative settings
            reserved_ratio = 0.15
            optimal_ratio = 0.75

        available_memory = total_memory_gb * (1 - reserved_ratio)
        optimal_memory = available_memory * optimal_ratio

        # Set reasonable bounds
        min_memory = 8.0
        max_memory = min(80.0, total_memory_gb * 0.9)  # Never exceed 90% of total

        optimal_memory = max(min_memory, min(optimal_memory, max_memory))

        mode_name = quantization_mode

        print(
            f"GPU Memory: {total_memory_gb:.1f}GB total, using {optimal_memory:.1f}GB for {mode_name} quantization"
        )

        return optimal_memory

    except Exception as e:
        print(f"Error calculating optimal memory, using default: {e}")
        return 24.0


class BagelModelLoader:
    """
    Unified BAGEL Model Loader with Dynamic Quantization Support

    Supports both standard BAGEL and DFloat11 models with optional quantization
    for ByteDance-Seed/BAGEL-7B-MoT model.
    """

    SUPPORTED_MODEL_REPOS = [
        "ByteDance-Seed/BAGEL-7B-MoT",
        "DFloat11/BAGEL-7B-MoT-DF11",
        "Yejy53/Echo-4o",
        "iitolstykh/Bagel-NHR-Edit",
        "iitolstykh/Bagel-NHR-Edit-V2",
        "sanaka87/BAGEL-RecA",
        "theunlikely/BAGEL-RecA-DF11",
        "multimodal-reasoning-lab/Bagel-Zebra-CoT",
    ]

    QUANTIZATION_MODES = {
        "BF16": "Standard (FP16/BF16)",
        "NF4": "NF4 (4-bit Quantization)",
        "INT8": "INT8 (8-bit Quantization)",
        "FP8": "FP8 (8-bit float)",
    }

    @classmethod
    def INPUT_TYPES(cls):
        # Discover local models and expose them as local:<name> choices
        discovered = discover_bagel_model_dirs()
        # show local folder names (e.g. "BAGEL-7B-MoT") and supported remote repo ids
        local_choices = [name for name in discovered.keys()]

        choices = local_choices + cls.SUPPORTED_MODEL_REPOS

        default_choice = local_choices[0] if len(local_choices) > 0 else cls.SUPPORTED_MODEL_REPOS[0]

        inputs = {
            "required": {
                "model_path": (
                    choices,
                    {
                        "default": default_choice,
                        "tooltip": "Select a local folder under models/bagel (folder name) or a supported remote repository id",
                    },
                ),
                # If True, will attempt to download remote repo when local folder is missing
                "allow_auto_download": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": "Allow automatic download of supported remote model if local folder is missing",
                    },
                ),
            }
        }

        inputs["required"]["quantization_mode"] = (
            list(cls.QUANTIZATION_MODES.keys()),
            {
                "default": "BF16",
                "tooltip": "Quantization: BF16=Standard, NF4=4-bit, INT8=8-bit, FP8=float8 (Only for ByteDance model)",
            },
        )

        return inputs

    RETURN_TYPES = ("BAGEL_MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "load_model"
    CATEGORY = "BAGEL/Deprecated"

    @classmethod
    def VALIDATE_INPUTS(cls, model_path, quantization_mode="BF16", **kwargs):
        """Validate input parameters. Supports local:<name> and remote repo ids.

        kwargs may contain allow_auto_download (ignored for validation).
        """
        discovered = discover_bagel_model_dirs()

        # Validate quantization_mode
        if quantization_mode not in cls.QUANTIZATION_MODES:
            return f"Invalid quantization_mode: {quantization_mode}. Supported: {list(cls.QUANTIZATION_MODES.keys())}"

        # Determine whether selected model implies DF11 or Echo-4o
        final_is_df11 = False
        # local folder selected (just the folder name)
        if isinstance(model_path, str) and model_path in discovered:
            name = model_path
            final_is_df11 = is_df11_name(name)
        # remote selection must be one of supported repos
        elif isinstance(model_path, str) and model_path in cls.SUPPORTED_MODEL_REPOS:
            final_is_df11 = is_df11_name(model_path)
        else:
            return f"Invalid model selection: {model_path}. Choose a local folder under models/bagel or a supported remote repo: {cls.SUPPORTED_MODEL_REPOS}"

        if final_is_df11:
            try:
                import dfloat11  # noqa: F401
            except ImportError:
                return (
                    "DFloat11 model selected, but the 'dfloat11' package is not "
                    "installed (it requires CUDA + cupy). Install it on a CUDA host "
                    "with: pip install dfloat11 --no-build-isolation  (or select a "
                    "standard BF16 BAGEL model instead)."
                )

        # If choosing quantization modes that require bitsandbytes for standard models
        if quantization_mode in ["NF4", "INT8"] and not final_is_df11:
            try:
                if importlib.util.find_spec("bitsandbytes") is None:
                    return f"Quantization mode {quantization_mode} requires bitsandbytes. Please install: pip install bitsandbytes"
            except Exception:
                return f"Quantization mode {quantization_mode} requires bitsandbytes. Please install: pip install bitsandbytes"

        # All validations passed
        return True

    def load_model(
        self,
        model_path: str,
        quantization_mode: str = "BF16",
        allow_auto_download: bool = False,
    ) -> Tuple[Dict[str, Any]]:
        """
        Load BAGEL model with unified interface supporting both standard and DFloat11 models.
        Quantization is applied only to ByteDance-Seed/BAGEL-7B-MoT model.
        """
        try:

            discovered = discover_bagel_model_dirs()

            # Resolve local_model_dir and model type flags
            is_df11_model = False
            is_echo4o_model = False
            local_model_dir = None
            # If a local folder name is selected
            if isinstance(model_path, str) and model_path in discovered:
                repo_name_segment = model_path
                local_model_dir = discovered[repo_name_segment]
                is_df11_model = is_df11_name(repo_name_segment)
                is_echo4o_model = is_echo4o_name(repo_name_segment)
            else:
                # remote selection must be in SUPPORTED_MODEL_REPOS
                if model_path not in self.SUPPORTED_MODEL_REPOS:
                    raise FileNotFoundError(f"Unsupported remote model selection: {model_path}. Supported: {self.SUPPORTED_MODEL_REPOS}")
                repo_name_segment = model_path.split("/")[-1]
                local_model_dir = os.path.join(comfy_models_dir, "bagel", repo_name_segment)
                is_df11_model = is_df11_name(model_path)
                is_echo4o_model = is_echo4o_name(model_path)

            if is_df11_model and quantization_mode != "BF16":
                print(
                    f"DFloat11 model detected. Ignoring quantization_mode {quantization_mode} and using built-in quantization."
                )
                quantization_mode = "BF16"

            # Echo-4o models use same configuration as standard BAGEL models
            if is_echo4o_model:
                print(f"Echo-4o model detected: {model_path} (Enhanced BAGEL with multi-image support)")

            common_vae_dir = os.path.join(comfy_models_dir, "vae")

            model_type = "DFloat11" if is_df11_model else ("Echo-4o" if is_echo4o_model else "BAGEL")
            print(
                f"Loading {model_type} model: {model_path} -> {local_model_dir} with {self.QUANTIZATION_MODES[quantization_mode]} mode..."
            )

            # Fix any existing nested directory structure
            fix_nested_model_structure(local_model_dir, model_path)

            if not os.path.exists(local_model_dir) or not check_model_files(local_model_dir, is_df11_model):
                # If the selection was a local folder, do not auto-download
                if isinstance(model_path, str) and model_path in discovered:
                    raise FileNotFoundError(f"Local model {model_path} missing required files in {local_model_dir}")

                # For remote repos, only attempt download if user allowed it
                if not allow_auto_download:
                    raise FileNotFoundError(
                        f"Model files for {model_path} not found in {local_model_dir}. Enable allow_auto_download=True to allow automatic download."
                    )

                print(f"Core model files not found for {model_path} at {local_model_dir}. allow_auto_download=True -> attempting download...")

                downloaded_path = download_model_with_hf_hub(local_model_dir, repo_id=model_path)
                if not downloaded_path:
                    raise FileNotFoundError(
                        f"Failed to download BAGEL model. Please manually download it from {model_path} and place it in {local_model_dir}"
                    )

                print(f"Successfully downloaded BAGEL model to {local_model_dir}")

                # Check for nested directory structure and fix if needed
                repo_name = model_path.split("/")[-1]
                nested_path = os.path.join(local_model_dir, repo_name)

                if os.path.exists(nested_path) and os.path.isdir(nested_path):
                    # Check if main directory is empty except for the nested folder
                    main_files = [f for f in os.listdir(local_model_dir) if f != repo_name and not f.startswith('.')]
                    nested_files = os.listdir(nested_path)

                    if len(main_files) == 0 and len(nested_files) > 0:
                        print(f"Detected nested folder structure. Moving files from {nested_path} to {local_model_dir}")

                        # Move all files from nested directory to parent directory
                        import shutil
                        for item in nested_files:
                            src = os.path.join(nested_path, item)
                            dst = os.path.join(local_model_dir, item)
                            if os.path.exists(dst):
                                if os.path.isdir(dst):
                                    shutil.rmtree(dst)
                                else:
                                    os.remove(dst)
                            shutil.move(src, dst)

                        # Remove the now-empty nested directory
                        os.rmdir(nested_path)
                        print(f"Successfully flattened directory structure for {model_path}")

            if not check_model_files(local_model_dir, is_df11_model):
                raise FileNotFoundError(
                    f"Required model files missing in {local_model_dir}"
                )

            # Load configuration files
            llm_config = Qwen2Config.from_json_file(
                os.path.join(local_model_dir, "llm_config.json")
            )
            llm_config.qk_norm = True
            llm_config.tie_word_embeddings = False
            llm_config.layer_module = "Qwen2MoTDecoderLayer"

            vit_config = SiglipVisionConfig.from_json_file(
                os.path.join(local_model_dir, "vit_config.json")
            )
            vit_config.rope = False
            vit_config.num_hidden_layers -= 1

            # Load VAE model using unified function
            vae_model, vae_config, vae_loaded_path = load_vae_model(local_model_dir)

            if is_df11_model:
                config = BagelConfig(
                    visual_gen=True,
                    visual_und=True,
                    llm_config=llm_config,
                    vit_config=vit_config,
                    vae_config=vae_config,
                    vit_max_num_patch_per_side=70,
                    connector_act="gelu_pytorch_tanh",
                    latent_patch_size=2,
                    max_latent_size=64,
                )
                with init_empty_weights():
                    language_model = Qwen2ForCausalLM(llm_config)
                    vit_model = SiglipVisionModel(vit_config)
                    model = Bagel(language_model, vit_model, config)
                    model.vit_model.vision_model.embeddings.convert_conv2d_to_linear(
                        vit_config, meta=True
                    )
                tokenizer = Qwen2Tokenizer.from_pretrained(local_model_dir)
                tokenizer, new_token_ids, _ = add_special_tokens(tokenizer)
                vae_transform = ImageTransform(1024, 512, 16)
                vit_transform = ImageTransform(980, 224, 14)
                model = model.to(torch.bfloat16)
                model.load_state_dict(
                    {
                        name: (
                            torch.empty(param.shape, dtype=param.dtype, device="cpu")
                            if param.device.type == "meta"
                            else param
                        )
                        for name, param in model.state_dict().items()
                    },
                    assign=True,
                )
                DFloat11Model = _require_dfloat11()
                model = DFloat11Model.from_pretrained(
                    local_model_dir,
                    bfloat16_model=model,
                    device="cpu",
                )

                # Create offload directory for model dispatch
                offload_dir = os.path.join(local_model_dir, "offload")
                os.makedirs(offload_dir, exist_ok=True)

                device_map = infer_auto_device_map(
                    model,
                    max_memory={0: f"{calculate_optimal_memory_gb('BF16')}GiB"},
                    no_split_module_classes=[
                        "Bagel",
                        "Qwen2MoTDecoderLayer",
                        "SiglipVisionModel",
                    ],
                )
                same_device_modules = [
                    "language_model.model.embed_tokens",
                    "time_embedder",
                    "latent_pos_embed",
                    "vae2llm",
                    "llm2vae",
                    "connector",
                    "vit_pos_embed",
                ]
                if torch.cuda.device_count() == 1:
                    first_device = device_map.get(same_device_modules[0], "cuda:0")
                    for k in same_device_modules:
                        if k in device_map:
                            device_map[k] = first_device
                        else:
                            device_map[k] = "cuda:0"
                else:
                    # Get first device with fallback to cuda:0
                    first_device = device_map.get(same_device_modules[0])
                    if first_device is None:
                        for device in device_map.values():
                            if isinstance(device, str) and device.startswith("cuda"):
                                first_device = device
                                break
                        else:
                            first_device = "cuda:0"

                    for k in same_device_modules:
                        device_map[k] = first_device

                model = dispatch_model(
                    model,
                    device_map=device_map,
                    offload_dir=offload_dir,
                    force_hooks=True,
                )
                model = model.eval()
                inferencer = InterleaveInferencer(
                    model=model,
                    vae_model=vae_model,
                    tokenizer=tokenizer,
                    vae_transform=vae_transform,
                    vit_transform=vit_transform,
                    new_token_ids=new_token_ids,
                )
                model_dict = {
                    "model": model,
                    "inferencer": inferencer,
                    "tokenizer": tokenizer,
                    "vae_model": vae_model,
                    "vae_transform": vae_transform,
                    "vit_transform": vit_transform,
                    "config": config,
                    "model_path": local_model_dir,
                    "model_repo_id": model_path,
                    "is_df11": is_df11_model,
                    "is_echo4o": is_echo4o_model,
                    "model_type": "DFloat11" if is_df11_model else ("Echo-4o" if is_echo4o_model else "BAGEL"),
                    "device": "cuda" if torch.cuda.is_available() else "cpu",
                }
                print(f"Successfully loaded BAGEL DF11 model from {local_model_dir}")
                return (model_dict,)
            elif not is_df11_model:
                # Create BAGEL configuration
                config = BagelConfig(
                    visual_gen=True,
                    visual_und=True,
                    llm_config=llm_config,
                    vit_config=vit_config,
                    vae_config=vae_config,
                    vit_max_num_patch_per_side=70,
                    connector_act="gelu_pytorch_tanh",
                    latent_patch_size=2,
                    max_latent_size=64,
                )

                # Initialize empty model
                with init_empty_weights():
                    language_model = Qwen2ForCausalLM(llm_config)
                    vit_model = SiglipVisionModel(vit_config)
                    model = Bagel(language_model, vit_model, config)
                    model.vit_model.vision_model.embeddings.convert_conv2d_to_linear(
                        vit_config, meta=True
                    )

                tokenizer = Qwen2Tokenizer.from_pretrained(local_model_dir)
                tokenizer, new_token_ids, _ = add_special_tokens(tokenizer)

                vae_transform = ImageTransform(1024, 512, 16)
                vit_transform = ImageTransform(980, 224, 14)

                if quantization_mode in ["NF4", "INT8", "FP8"]:
                    max_memory_gb = calculate_optimal_memory_gb(quantization_mode)
                    if quantization_mode == "FP8":
                        max_memory_gb *= 4 # Optimization doesn't know about quantization
                    max_memory_per_gpu = f"{max_memory_gb}GiB"
                    device_map = infer_auto_device_map(
                        model,
                        max_memory={
                            i: max_memory_per_gpu
                            for i in range(torch.cuda.device_count())
                        },
                        no_split_module_classes=["Bagel", "Qwen2MoTDecoderLayer"],
                    )
                else:
                    device_map = infer_auto_device_map(
                        model,
                        no_split_module_classes=["Bagel", "Qwen2MoTDecoderLayer"],
                    )

                # Ensure same device placement for critical modules
                same_device_modules = [
                    "language_model.model.embed_tokens",
                    "time_embedder",
                    "latent_pos_embed",
                    "vae2llm",
                    "llm2vae",
                    "connector",
                    "vit_pos_embed",
                ]

                if torch.cuda.device_count() == 1:
                    first_device = device_map.get(same_device_modules[0], "cuda:0")
                    for k in same_device_modules:
                        if k in device_map:
                            device_map[k] = first_device
                        else:
                            device_map[k] = "cuda:0"
                else:
                    first_device = device_map.get(same_device_modules[0])
                    if first_device is None:
                        for device in device_map.values():
                            if isinstance(device, str) and device.startswith("cuda"):
                                first_device = device
                                break
                        else:
                            first_device = "cuda:0"

                    for k in same_device_modules:
                        if k in device_map:
                            device_map[k] = first_device
                        else:
                            device_map[k] = first_device

                # Load model based on quantization mode
                checkpoint_path = os.path.join(local_model_dir, "ema.safetensors")

                if quantization_mode == "BF16":
                    # Standard loading
                    print("Loading model in standard FP16/BF16 mode...")
                    model = load_checkpoint_and_dispatch(
                        model,
                        checkpoint=checkpoint_path,
                        device_map=device_map,
                        offload_buffers=True,
                        offload_folder="offload",
                        dtype=torch.bfloat16,
                        force_hooks=True,
                    ).eval()

                elif quantization_mode == "NF4":
                    print("Loading model with NF4 (4-bit) quantization...")
                    bnb_quantization_config = BnbQuantizationConfig(
                        load_in_4bit=True,
                        bnb_4bit_compute_dtype=torch.bfloat16,
                        bnb_4bit_use_double_quant=False,
                        bnb_4bit_quant_type="nf4",
                    )
                    model = load_and_quantize_model(
                        model,
                        weights_location=checkpoint_path,
                        bnb_quantization_config=bnb_quantization_config,
                        device_map=device_map,
                        offload_folder="offload",
                    ).eval()

                elif quantization_mode == "INT8":
                    print("Loading model with INT8 (8-bit) quantization...")
                    bnb_quantization_config = BnbQuantizationConfig(
                        load_in_8bit=True, torch_dtype=torch.bfloat16
                    )
                    model = load_and_quantize_model(
                        model,
                        weights_location=checkpoint_path,
                        bnb_quantization_config=bnb_quantization_config,
                        device_map=device_map,
                        offload_folder="offload",
                    ).eval()

                elif quantization_mode == "FP8":
                    # 8-bit float
                    print("Loading model in FP8 mode...")
                    model = load_checkpoint_and_dispatch(
                        model,
                        checkpoint=checkpoint_path,
                        device_map=device_map,
                        offload_buffers=True,
                        offload_folder="offload",
                        dtype=torch.float8_e4m3fn,
                        force_hooks=True,
                    ).eval()

                else:
                    raise ValueError(
                        f"Unsupported quantization mode: {quantization_mode}"
                    )

                # Create inferencer
                inferencer = InterleaveInferencer(
                    model=model,
                    vae_model=vae_model,
                    tokenizer=tokenizer,
                    vae_transform=vae_transform,
                    vit_transform=vit_transform,
                    new_token_ids=new_token_ids,
                )

                model_dict = {
                    "model": model,
                    "inferencer": inferencer,
                    "tokenizer": tokenizer,
                    "vae_model": vae_model,
                    "vae_transform": vae_transform,
                    "vit_transform": vit_transform,
                    "config": config,
                    "model_path": local_model_dir,
                    "model_repo_id": model_path,
                    "quantization_mode": quantization_mode,
                    "quantization_info": self.QUANTIZATION_MODES[quantization_mode],
                    "is_df11": False,
                    "is_echo4o": is_echo4o_model,
                    "model_type": "Echo-4o" if is_echo4o_model else "BAGEL",
                    "device": "cuda" if torch.cuda.is_available() else "cpu",
                }

                model_type_display = "Echo-4o" if is_echo4o_model else "BAGEL"
                print(
                    f"Successfully loaded {model_type_display} model with {self.QUANTIZATION_MODES[quantization_mode]} mode"
                )
                return (model_dict,)

        except Exception as e:
            print(f"Error loading BAGEL model: {e}")
            raise e


class BagelTextToImage:
    """BAGEL Text to Image Node"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("BAGEL_MODEL", {"tooltip": "BAGEL model"}),
                "prompt": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "A female cosplayer portraying an ethereal fairy or elf, wearing a flowing dress made of delicate fabrics in soft, mystical colors like emerald green and silver. She has pointed ears, a gentle, enchanting expression, and her outfit is adorned with sparkling jewels and intricate patterns. The background is a magical forest with glowing plants, mystical creatures, and a serene atmosphere.",
                        "tooltip": "Text prompt",
                    },
                ),
                "seed": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 1000000,
                        "tooltip": "Random seed, 0 for random",
                    },
                ),
                "image_ratio": (
                    ["1:1", "4:3", "3:4", "16:9", "9:16"],
                    {"default": "1:1", "tooltip": "Image aspect ratio"},
                ),
                "cfg_text_scale": (
                    "FLOAT",
                    {
                        "default": 4.0,
                        "min": 1.0,
                        "max": 8.0,
                        "step": 0.1,
                        "tooltip": "CFG text scaling",
                    },
                ),
                "num_timesteps": (
                    "INT",
                    {
                        "default": 50,
                        "min": 10,
                        "max": 100,
                        "step": 5,
                        "tooltip": "Denoising steps",
                    },
                ),
            },
            "optional": {
                "show_thinking": (
                    "BOOLEAN",
                    {"default": False, "tooltip": "Display reasoning process"},
                ),
                "cfg_interval": (
                    "FLOAT",
                    {
                        "default": 0.4,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.1,
                        "tooltip": "CFG interval start value",
                    },
                ),
                "timestep_shift": (
                    "FLOAT",
                    {
                        "default": 3.0,
                        "min": 1.0,
                        "max": 5.0,
                        "step": 0.5,
                        "tooltip": "Timestep offset",
                    },
                ),
                "cfg_renorm_min": (
                    "FLOAT",
                    {
                        "default": 0.0,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.1,
                        "tooltip": "CFG re-normalization minimum value",
                    },
                ),
                "cfg_renorm_type": (
                    ["global", "local", "text_channel"],
                    {"default": "global", "tooltip": "CFG re-normalization type"},
                ),
                "text_temperature": (
                    "FLOAT",
                    {
                        "default": 0.3,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.1,
                        "tooltip": "Text generation temperature",
                    },
                ),
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("image", "thinking")
    FUNCTION = "generate_image"
    CATEGORY = "BAGEL/Deprecated"

    @classmethod
    def VALIDATE_INPUTS(
        cls, model, prompt, seed, image_ratio, cfg_text_scale, num_timesteps, **kwargs
    ):
        """Validate input parameters"""
        if not validate_seed(seed):
            return "Seed must be a non-negative integer"

        if image_ratio not in ["1:1", "4:3", "3:4", "16:9", "9:16"]:
            return f"Invalid image_ratio: {image_ratio}"

        if not validate_cfg_scale(cfg_text_scale):
            return "cfg_text_scale must be between 1.0 and 8.0"

        if not validate_timesteps(num_timesteps):
            return "num_timesteps must be between 10 and 100"

        return True

    def generate_image(
        self,
        model: Dict[str, Any],
        prompt: str,
        seed: int,
        image_ratio: str,
        cfg_text_scale: float,
        num_timesteps: int,
        show_thinking: bool = False,
        cfg_interval: float = 0.4,
        timestep_shift: float = 3.0,
        cfg_renorm_min: float = 0.0,
        cfg_renorm_type: str = "global",
        text_temperature: float = 0.3,
    ) -> Tuple[torch.Tensor, str]:
        """
        Generate image from text using BAGEL model

        Args:
            model: BAGEL model dictionary
            prompt: Text prompt
            seed: Random seed
            image_ratio: Image aspect ratio
            cfg_text_scale: CFG text scaling
            num_timesteps: Denoising steps
            show_thinking: Whether to display the reasoning process
            cfg_interval: CFG interval start value
            timestep_shift: Timestep offset
            cfg_renorm_min: CFG re-normalization minimum value
            cfg_renorm_type: CFG re-normalization type
            text_temperature: Text generation temperature

        Returns:
            Generated image tensor and reasoning process text
        """
        try:
            # Set random seed
            set_seed(seed)

            # Get inferencer
            inferencer = model["inferencer"]

            # Set image dimensions
            image_shapes_map = get_image_shapes_map()
            image_shapes = image_shapes_map[image_ratio]

            # Set inference hyperparameters
            inference_hyper = create_common_inference_config(
                cfg_text_scale=cfg_text_scale,
                num_timesteps=num_timesteps,
                cfg_interval=cfg_interval,
                timestep_shift=timestep_shift,
                cfg_renorm_min=cfg_renorm_min,
                cfg_renorm_type=cfg_renorm_type,
                text_temperature=text_temperature,
                show_thinking=show_thinking,
                image_shapes=image_shapes,
            )

            # Initialize ProgressBar
            pbar = setup_inference_progress_bar(num_timesteps)

            # Call inferencer, passing pbar
            result = inferencer(
                text=prompt, think=show_thinking, pbar=pbar, **inference_hyper
            )

            # Convert image format
            pil_image = result["image"]
            tensor_image = pil_to_tensor(pil_image)

            # Get reasoning process
            thinking_text = result.get("text", "") if show_thinking else ""

            print(f"Generated image with size: {pil_image.size}")

            return (tensor_image, thinking_text)

        except Exception as e:
            print(f"Error in text to image generation: {e}")
            # Return empty image and error message
            return (create_error_response_image(), f"Error: {str(e)}")


class BagelImageEdit:
    """BAGEL Image Edit Node"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("BAGEL_MODEL", {"tooltip": "BAGEL model"}),
                "image": ("IMAGE", {"tooltip": "Input image"}),
                "prompt": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "Edit the image according to the description",
                        "tooltip": "Editing prompt",
                    },
                ),
                "seed": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 1000000,
                        "tooltip": "Random seed, 0 for random",
                    },
                ),
                "cfg_text_scale": (
                    "FLOAT",
                    {
                        "default": 4.0,
                        "min": 1.0,
                        "max": 8.0,
                        "step": 0.1,
                        "tooltip": "CFG text scaling",
                    },
                ),
                "cfg_img_scale": (
                    "FLOAT",
                    {
                        "default": 2.0,
                        "min": 1.0,
                        "max": 4.0,
                        "step": 0.1,
                        "tooltip": "CFG image scaling",
                    },
                ),
                "num_timesteps": (
                    "INT",
                    {
                        "default": 50,
                        "min": 10,
                        "max": 100,
                        "step": 5,
                        "tooltip": "Denoising steps",
                    },
                ),
            },
            "optional": {
                "show_thinking": (
                    "BOOLEAN",
                    {"default": False, "tooltip": "Display reasoning process"},
                ),
                "cfg_interval": (
                    "FLOAT",
                    {
                        "default": 0.0,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.1,
                        "tooltip": "CFG interval start value",
                    },
                ),
                "timestep_shift": (
                    "FLOAT",
                    {
                        "default": 3.0,
                        "min": 1.0,
                        "max": 10.0,
                        "step": 0.5,
                        "tooltip": "Timestep offset",
                    },
                ),
                "cfg_renorm_min": (
                    "FLOAT",
                    {
                        "default": 0.0,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.1,
                        "tooltip": "CFG re-normalization minimum value",
                    },
                ),
                "cfg_renorm_type": (
                    ["global", "local", "text_channel"],
                    {"default": "text_channel", "tooltip": "CFG re-normalization type"},
                ),
                "text_temperature": (
                    "FLOAT",
                    {
                        "default": 0.3,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.1,
                        "tooltip": "Text generation temperature",
                    },
                ),
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("image", "thinking")
    FUNCTION = "edit_image"
    CATEGORY = "BAGEL/Deprecated"

    @classmethod
    def VALIDATE_INPUTS(
        cls,
        model,
        image,
        prompt,
        seed,
        cfg_text_scale,
        cfg_img_scale,
        num_timesteps,
        **kwargs,
    ):
        if not validate_seed(seed):
            return "Seed must be a non-negative integer"

        if not validate_cfg_scale(cfg_text_scale):
            return "cfg_text_scale must be between 1.0 and 8.0"

        if not validate_cfg_scale(cfg_img_scale, min_val=1.0, max_val=4.0):
            return "cfg_img_scale must be between 1.0 and 4.0"

        if not validate_timesteps(num_timesteps):
            return "num_timesteps must be between 10 and 100"

        return True

    def edit_image(
        self,
        model: Dict[str, Any],
        image: torch.Tensor,
        prompt: str,
        seed: int,
        cfg_text_scale: float,
        cfg_img_scale: float,
        num_timesteps: int,
        show_thinking: bool = False,
        cfg_interval: float = 0.0,
        timestep_shift: float = 3.0,
        cfg_renorm_min: float = 0.0,
        cfg_renorm_type: str = "text_channel",
        text_temperature: float = 0.3,
    ) -> Tuple[torch.Tensor, str]:
        """
        Edit image using BAGEL model

        Args:
            model: BAGEL model dictionary
            image: Input image tensor
            prompt: Editing prompt
            seed: Random seed
            cfg_text_scale: CFG text scaling
            cfg_img_scale: CFG image scaling
            num_timesteps: Denoising steps
            show_thinking: Whether to display the reasoning process
            cfg_interval: CFG interval start value
            timestep_shift: Timestep offset
            cfg_renorm_min: CFG re-normalization minimum value
            cfg_renorm_type: CFG re-normalization type
            text_temperature: Text generation temperature

        Returns:
            Edited image tensor and reasoning process text
        """
        try:
            # Set random seed
            set_seed(seed)

            # Get inferencer
            inferencer = model["inferencer"]

            # Convert image format
            pil_image = tensor_to_pil(image)
            pil_image = pil_img2rgb(pil_image)

            # Set inference hyperparameters
            inference_hyper = create_common_inference_config(
                cfg_text_scale=cfg_text_scale,
                num_timesteps=num_timesteps,
                cfg_interval=cfg_interval,
                timestep_shift=timestep_shift,
                cfg_renorm_min=cfg_renorm_min,
                cfg_renorm_type=cfg_renorm_type,
                text_temperature=text_temperature,
                show_thinking=show_thinking,
                cfg_img_scale=cfg_img_scale,
            )

            # Initialize ProgressBar
            pbar = setup_inference_progress_bar(num_timesteps)

            # Call inferencer, passing pbar
            result = inferencer(
                image=pil_image,
                text=prompt,
                think=show_thinking,
                pbar=pbar,
                **inference_hyper,
            )

            # Convert image format
            edited_pil_image = result["image"]
            tensor_image = pil_to_tensor(edited_pil_image)

            # Get reasoning process
            thinking_text = result.get("text", "") if show_thinking else ""

            print(f"Edited image with size: {edited_pil_image.size}")

            return (tensor_image, thinking_text)

        except Exception as e:
            print(f"Error in image editing: {e}")
            # Return original image and error message
            return (image, f"Error: {str(e)}")


class BagelImageUnderstanding:
    """BAGEL Image Understanding Node"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("BAGEL_MODEL", {"tooltip": "BAGEL model"}),
                "image": ("IMAGE", {"tooltip": "Input image"}),
                "prompt": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "What do you see in this image?",
                        "tooltip": "Question text",
                    },
                ),
            },
            "optional": {
                "show_thinking": (
                    "BOOLEAN",
                    {"default": False, "tooltip": "Display reasoning process"},
                ),
                "do_sample": (
                    "BOOLEAN",
                    {"default": False, "tooltip": "Enable sampling"},
                ),
                "text_temperature": (
                    "FLOAT",
                    {
                        "default": 0.3,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.05,
                        "tooltip": "Text generation temperature",
                    },
                ),
                "max_new_tokens": (
                    "INT",
                    {
                        "default": 512,
                        "min": 64,
                        "max": 4096,
                        "step": 64,
                        "tooltip": "Maximum new tokens",
                    },
                ),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    FUNCTION = "understand_image"
    CATEGORY = "BAGEL/Deprecated"

    @classmethod
    def VALIDATE_INPUTS(cls, model, image, prompt, **kwargs):
        """Validate input parameters"""
        # Validate optional parameters
        if "text_temperature" in kwargs:
            if not validate_temperature(kwargs["text_temperature"]):
                return "text_temperature must be between 0.0 and 1.0"

        if "max_new_tokens" in kwargs:
            if not validate_max_tokens(kwargs["max_new_tokens"]):
                return "max_new_tokens must be between 64 and 4096"

        return True

    def understand_image(
        self,
        model: Dict[str, Any],
        image: torch.Tensor,
        prompt: str,
        show_thinking: bool = False,
        do_sample: bool = False,
        text_temperature: float = 0.3,
        max_new_tokens: int = 512,
    ) -> Tuple[str]:
        """
        Use BAGEL model to understand image and answer questions

        Args:
            model: BAGEL model dictionary
            image: Input image tensor
            prompt: Question text
            show_thinking: Whether to display the reasoning process
            do_sample: Whether to enable sampling
            text_temperature: Text generation temperature
            max_new_tokens: Maximum new tokens

        Returns:
            Answer text
        """
        try:
            # Get inferencer
            inferencer = model["inferencer"]

            # Convert image format
            pil_image = tensor_to_pil(image)
            pil_image = pil_img2rgb(pil_image)

            # Set inference hyperparameters
            inference_hyper = {
                "do_sample": do_sample,
                "text_temperature": text_temperature,
                "max_think_token_n": max_new_tokens,
            }

            # Call inferencer
            result = inferencer(
                image=pil_image,
                text=prompt,
                think=show_thinking,
                understanding_output=True,
                **inference_hyper,
            )

            answer_text = result["text"]

            print(f"Image understanding completed, response length: {len(answer_text)}")

            return (answer_text,)

        except Exception as e:
            print(f"Error in image understanding: {e}")
            return (f"Error: {str(e)}",)


class BagelMultiImageEdit:
    """BAGEL Multi-Image Edit Node - Enhanced for Echo-4o (2-4 reference images)"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("BAGEL_MODEL", {"tooltip": "BAGEL/Echo-4o model"}),
                "ref_image_1": ("IMAGE", {"tooltip": "First reference image (required)"}),
                "ref_image_2": ("IMAGE", {"tooltip": "Second reference image (required)"}),
                "prompt": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "Combine elements from the reference images according to the description.",
                        "tooltip": "Multi-image editing prompt",
                    },
                ),
                "seed": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 1000000,
                        "tooltip": "Random seed, 0 for random",
                    },
                ),
                "cfg_text_scale": (
                    "FLOAT",
                    {
                        "default": 4.0,
                        "min": 1.0,
                        "max": 8.0,
                        "step": 0.1,
                        "tooltip": "CFG text scaling",
                    },
                ),
                "cfg_img_scale": (
                    "FLOAT",
                    {
                        "default": 2.0,
                        "min": 1.0,
                        "max": 3.0,
                        "step": 0.1,
                        "tooltip": "CFG image scaling",
                    },
                ),
                "num_timesteps": (
                    "INT",
                    {
                        "default": 50,
                        "min": 10,
                        "max": 100,
                        "step": 5,
                        "tooltip": "Denoising steps",
                    },
                ),
            },
            "optional": {
                "ref_image_3": ("IMAGE", {"tooltip": "Third reference image (optional)"}),
                "ref_image_4": ("IMAGE", {"tooltip": "Fourth reference image (optional)"}),
                "show_thinking": (
                    "BOOLEAN",
                    {"default": False, "tooltip": "Display reasoning process"},
                ),
                "cfg_interval": (
                    "FLOAT",
                    {
                        "default": 0.0,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.1,
                        "tooltip": "CFG interval start value",
                    },
                ),
                "timestep_shift": (
                    "FLOAT",
                    {
                        "default": 3.0,
                        "min": 1.0,
                        "max": 5.0,
                        "step": 0.5,
                        "tooltip": "Timestep offset",
                    },
                ),
                "cfg_renorm_min": (
                    "FLOAT",
                    {
                        "default": 0.0,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.1,
                        "tooltip": "CFG re-normalization minimum value",
                    },
                ),
                "cfg_renorm_type": (
                    ["global", "local", "text_channel"],
                    {"default": "text_channel", "tooltip": "CFG re-normalization type"},
                ),
                "text_temperature": (
                    "FLOAT",
                    {
                        "default": 0.3,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.1,
                        "tooltip": "Text generation temperature",
                    },
                ),
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("image", "thinking")
    FUNCTION = "edit_multi_images"
    CATEGORY = "BAGEL/Deprecated"

    @classmethod
    def VALIDATE_INPUTS(
        cls, model, ref_image_1, ref_image_2, prompt, seed, cfg_text_scale, cfg_img_scale, num_timesteps, **kwargs
    ):
        """Validate input parameters"""
        if not validate_seed(seed):
            return "Invalid seed value"

        if not validate_cfg_scale(cfg_text_scale):
            return "Invalid cfg_text_scale value"

        if not validate_cfg_scale(cfg_img_scale, min_val=1.0, max_val=3.0):
            return "Invalid cfg_img_scale value"

        if not validate_timesteps(num_timesteps):
            return "Invalid num_timesteps value"

        return True

    def edit_multi_images(
        self,
        model: Dict[str, Any],
        ref_image_1: torch.Tensor,
        ref_image_2: torch.Tensor,
        prompt: str,
        seed: int,
        cfg_text_scale: float,
        cfg_img_scale: float,
        num_timesteps: int,
        ref_image_3: Optional[torch.Tensor] = None,
        ref_image_4: Optional[torch.Tensor] = None,
        show_thinking: bool = False,
        cfg_interval: float = 0.0,
        timestep_shift: float = 3.0,
        cfg_renorm_min: float = 0.0,
        cfg_renorm_type: str = "text_channel",
        text_temperature: float = 0.3,
    ) -> Tuple[torch.Tensor, str]:
        """
        Edit multiple images using BAGEL/Echo-4o model

        Args:
            model: BAGEL model dictionary
            ref_image_1: First reference image (required)
            ref_image_2: Second reference image (required)
            prompt: Multi-image editing prompt
            seed: Random seed
            cfg_text_scale: CFG text scaling
            cfg_img_scale: CFG image scaling
            num_timesteps: Denoising steps
            ref_image_3: Third reference image (optional)
            ref_image_4: Fourth reference image (optional)
            show_thinking: Whether to display the reasoning process
            cfg_interval: CFG interval start value
            timestep_shift: Timestep offset
            cfg_renorm_min: CFG re-normalization minimum value
            cfg_renorm_type: CFG re-normalization type
            text_temperature: Text generation temperature

        Returns:
            Generated image tensor and reasoning process text
        """
        try:
            # Set random seed
            set_seed(seed)

            # Get inferencer
            inferencer = model["inferencer"]

            # Check if this is Echo-4o model
            is_echo4o = model.get("is_echo4o", False)
            model_type = model.get("model_type", "BAGEL")

            if not is_echo4o:
                print("Note: Multi-image editing is optimized for Echo-4o models. Current model may have limited multi-image support.")

            # Collect all reference images
            pil_images = []

            # Add required images
            pil_images.append(tensor_to_pil(ref_image_1))
            pil_images.append(tensor_to_pil(ref_image_2))

            # Add optional images if provided
            if ref_image_3 is not None:
                pil_images.append(tensor_to_pil(ref_image_3))
            if ref_image_4 is not None:
                pil_images.append(tensor_to_pil(ref_image_4))

            print(f"Processing {len(pil_images)} reference images with {model_type} model")

            # Set up progress bar
            pbar = setup_inference_progress_bar(num_timesteps)

            # Create inference configuration
            inference_hyper = create_common_inference_config(
                cfg_text_scale=cfg_text_scale,
                num_timesteps=num_timesteps,
                cfg_interval=cfg_interval,
                timestep_shift=timestep_shift,
                cfg_renorm_min=cfg_renorm_min,
                cfg_renorm_type=cfg_renorm_type,
                text_temperature=text_temperature,
                show_thinking=show_thinking,
                cfg_img_scale=cfg_img_scale,
            )

            print(f"Multi-image editing prompt: {prompt}")
            print("-" * 50)

            # Execute inference with multiple images (Echo-4o enhanced)
            result = inferencer(
                image=pil_images,  # pass list of images
                text=prompt,
                think=show_thinking,
                pbar=pbar,
                **inference_hyper,
            )

            generated_image = result["image"]
            thinking_text = result.get("text", "")

            if generated_image is None:
                print("Warning: No image generated")
                return (create_error_response_image(), thinking_text)

            # Convert PIL to tensor
            result_tensor = pil_to_tensor(generated_image)

            print(f"Multi-image editing completed with {len(pil_images)} references")
            if show_thinking and thinking_text:
                print(f"AI thinking process: {thinking_text[:200]}...")

            return (result_tensor, thinking_text)

        except Exception as e:
            print(f"Error in multi-image editing: {e}")
            return (create_error_response_image(), f"Error: {str(e)}")


# Node mappings
NODE_CLASS_MAPPINGS = {
    "BagelModelLoader": BagelModelLoader,
    "BagelTextToImage": BagelTextToImage,
    "BagelImageEdit": BagelImageEdit,
    "BagelImageUnderstanding": BagelImageUnderstanding,
    "BagelMultiImageEdit": BagelMultiImageEdit,
}

# Display name mappings
NODE_DISPLAY_NAME_MAPPINGS = {
    "BagelModelLoader": "BAGEL Model Loader (Deprecated)",
    "BagelTextToImage": "BAGEL Text to Image (Deprecated)",
    "BagelImageEdit": "BAGEL Image Edit (Deprecated)",
    "BagelImageUnderstanding": "BAGEL Image Understanding (Deprecated)",
    "BagelMultiImageEdit": "BAGEL Multi-Image Edit (Deprecated, Echo-4o only)",
}

# Export for ComfyUI
__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
