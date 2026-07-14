# ComfyUI-BAGEL

A ComfyUI custom node package for BAGEL-7B-MoT with native ComfyUI model loading.

## Native ComfyUI layout

```mermaid
flowchart LR
    A["6chan/bagel_comfy<br/>single-file BAGEL safetensors"] --> B["ComfyUI/models/diffusion_models"]
    C["FLUX AE<br/>ae.safetensors"] --> D["ComfyUI/models/vae"]
    E["Packaged Qwen tokenizer<br/>inside this custom node"] --> F["BAGEL Model Loader"]
    B --> F
    D --> G["Official VAELoader / VAEEncode / VAEDecode"]
    F --> H["BAGEL native nodes"]
    G --> H
```

| Component | Recommended source | Put it here | Loaded by |
| --- | --- | --- | --- |
| BAGEL main model | [`6chan/bagel_comfy`](https://huggingface.co/6chan/bagel_comfy) single-file `.safetensors` | `ComfyUI/models/diffusion_models/` | `BAGEL Model Loader` |
| FLUX AE / VAE | `ae.safetensors` for FLUX | `ComfyUI/models/vae/` | official `VAELoader`, `VAEEncode`, `VAEDecode` |
| Qwen tokenizer | bundled in this repository | no manual install | `BAGEL Model Loader` |
| BAGEL model configs | embedded in the converted `.safetensors` metadata | no manual install | `BAGEL Model Loader` |
| Old HF shard layout | `ByteDance-Seed/BAGEL-7B-MoT`, `DFloat11/BAGEL-7B-MoT-DF11` | `ComfyUI/models/bagel/` | deprecated legacy nodes |

> Recommended: download the converted single-file model from `6chan/bagel_comfy`.
> If you already have the original BAGEL checkpoint, either convert it with
> `scripts/convert_bagel_model.py` or re-download the converted file.
> If the Hugging Face repository also contains config files, treat them as
> conversion/audit references. The native ComfyUI loader does not require users
> to copy config files into `models/diffusion_models`.

## Install

| Step | Action |
| --- | --- |
| 1 | Clone this repository into `ComfyUI/custom_nodes/ComfyUI-BAGEL`. |
| 2 | Install Python dependencies with `pip install -r requirements.txt`. |
| 3 | Download a converted BAGEL `.safetensors` from [`6chan/bagel_comfy`](https://huggingface.co/6chan/bagel_comfy) into `ComfyUI/models/diffusion_models/`. |
| 4 | Put FLUX `ae.safetensors` into `ComfyUI/models/vae/`. |
| 5 | Restart ComfyUI and load one of the native workflows below. |

### Existing old-model users

| Current state | Recommended action |
| --- | --- |
| You already downloaded `ByteDance-Seed/BAGEL-7B-MoT` | Convert it with `scripts/convert_bagel_model.py`, or re-download the converted single-file model. |
| You already downloaded `DFloat11/BAGEL-7B-MoT-DF11` | Keep using deprecated workflows for now, or convert/re-download when a converted quantized release is available. |
| You have old all-in-one BAGEL workflows | Use the `_deprecated` workflow files and deprecated nodes, then migrate to native workflows. |

Legacy auto-download and all-in-one loader instructions were moved to
[`docs/deprecated-installation.md`](docs/deprecated-installation.md).

## Workflows

```mermaid
flowchart TB
    subgraph T2I["Text to image"]
        M1["BAGEL Model Loader"] --> T["BAGEL Text to Image"]
        V1["VAELoader ae.safetensors"] --> D1["VAEDecode"]
        T --> D1 --> S1["SaveImage"]
    end

    subgraph EDIT["Image editing"]
        I["LoadImage"] --> E["BAGEL Image Edit"]
        I --> VE["VAEEncode"]
        V2["VAELoader ae.safetensors"] --> VE
        VE --> E
        M2["BAGEL Model Loader"] --> E
        E --> D2["VAEDecode"] --> S2["SaveImage"]
        V2 --> D2
    end

    subgraph VQA["Image understanding"]
        I2["LoadImage"] --> U["BAGEL Image Understanding"]
        M3["BAGEL Model Loader"] --> U
        U --> TXT["Show Text / Preview as Text"]
    end
```

| Workflow | File | Extra nodes | Notes |
| --- | --- | --- | --- |
| Text-to-image | `example_workflows/bagel_text_to_image.json` | none | Native BAGEL latent generation, official `VAEDecode`. |
| Image editing | `example_workflows/bagel_image_editing.json` | none | Official `VAEEncode` feeds source-image latent conditioning; official `VAEDecode` decodes output. |
| Image understanding | `example_workflows/bagel_image_understanding.json` | `ShowText|pysssss` from `comfyui-custom-scripts`, or replace with official `Preview as Text` on newer ComfyUI | VIT/text path only; no VAE nodes required. |
| Deprecated text-to-image | `example_workflows/bagel_text_to_image_deprecated.json` | `comfyui-custom-scripts` | Old all-in-one loader. |
| Deprecated image editing | `example_workflows/bagel_image_editing_deprecated.json` | `comfyui-custom-scripts` | Old all-in-one loader. |
| Deprecated image understanding | `example_workflows/bagel_image_understanding_deprecated.json` | `comfyui-custom-scripts` | Old all-in-one loader. |

## Model and runtime matrix

| Path | Model source | File layout | Nodes | VAE | Status |
| --- | --- | --- | --- | --- | --- |
| Native BF16 | `6chan/bagel_comfy` | single `.safetensors` in `diffusion_models` | `BAGEL*` native nodes | official FLUX AE | recommended |
| Converted local BF16 | original `ByteDance-Seed/BAGEL-7B-MoT` converted by script | single `.safetensors` in `diffusion_models` | `BAGEL*` native nodes | official FLUX AE | supported |
| Legacy standard | original HF shard folder | folder in `models/bagel` | `Bagel* (Deprecated)` | internal legacy VAE | compatibility only |
| Legacy DFloat11 | DFloat11 HF folder | folder in `models/bagel` | `Bagel* (Deprecated)` | internal legacy VAE | compatibility only |

| Task | Recommended workflow | Expected VRAM | Current validation |
| --- | --- | --- | --- |
| Text-to-image | native BF16 + official VAE decode | A100-class / high-VRAM GPU recommended | ran on Modal A100; exact benchmark pending |
| Image editing | native BF16 + official VAE encode/decode | A100-class / high-VRAM GPU recommended | ran on Modal A100; exact benchmark pending |
| Image understanding | native BF16, no VAE | lower than generation/editing, still needs BAGEL loaded | workflow prepared; run on remote GPU |
| Legacy DFloat11 generation | deprecated workflows | 21.76 GB reported for 1024x1024 | old README reported 154.39 s on RTX 4090 |
| Legacy standard generation | deprecated workflows | 30.07 GB reported for 1024x1024 | old README reported 482.95 s on RTX 4090 |

## Future model support plan

Future BAGEL variants will be tracked from the
[`6chan/bagel`](https://huggingface.co/collections/6chan/bagel) collection. The
support path depends on each model's file format, modality inputs, and inference
logic.

```mermaid
flowchart LR
    C["6chan/bagel collection"] --> A["Same BAGEL runtime shape"]
    C --> B["Different conditioning / task logic"]
    C --> D["Different weight format"]
    A --> A1["Add metadata + reuse native nodes"]
    B --> B1["Add adapter or new task node"]
    D --> D1["Add converter / loader support"]
```

| Variant family | Example models in collection | Input / output shape | Planned ComfyUI support |
| --- | --- | --- | --- |
| Base BAGEL any-to-any | `ByteDance-Seed/BAGEL-7B-MoT`, `6chan/bagel_comfy` | text, image, latent -> text/image | current native nodes |
| Image editing / NHR editing | `iitolstykh/Bagel-NHR-Edit`, `Bagel-NHR-Edit-V2` | image + edit prompt -> image | first try native image-edit adapter; add a dedicated edit node if prompt/image order differs |
| Reasoning / VQA variants | `multimodal-reasoning-lab/Bagel-Zebra-CoT`, `sensenova/SenseNova-SI-1.1-BAGEL-7B-MoT` | image + text -> text | adapt `BAGEL Image Understanding`; add reasoning-specific controls if needed |
| Text-to-image variants | `Wayne-King/SRUM_BAGEL_7B_MoT`, `Ryann829/Scone`, `LLM-Drop/*GEN*`, `Yanran21/UniGenDet` | text -> image | adapt `BAGEL Text to Image`; add model-specific generation options only when required |
| Quantized formats | `DFloat11/*`, FP8, INT8, GGUF, AutoRound INT4 | same tasks, different weight format/runtime | separate loader/converter path; do not mix into the BF16 loader unless the state dict is compatible |
| Specialized any-to-any / composition | `ThinkMorph`, `Uni-Edit`, `UniCorn`, `ConsistCompose`, `Echo-4o`, `SenseNova-Vision` | may add multi-image, identity, composition, or agentic conditioning | inspect model card + sample code first; add new nodes when graph inputs differ from the base BAGEL tasks |

| Support stage | Acceptance gate |
| --- | --- |
| Catalog | Add model to a comparison table with source, license, task, file format, expected VRAM, and required nodes. |
| Load | Converted or downloaded model appears in the correct ComfyUI model folder and loads without auto-downloading pipeline code. |
| Smoke test | One minimal workflow runs on a remote GPU for the model's primary task. |
| Workflow | Add or update an example workflow with official ComfyUI nodes where possible. |
| Benchmark | Record GPU, VRAM peak, image size, steps, and wall-clock time. |

## Free / low-cost GPU candidates for validation

These are candidates for installation/import checks or small smoke tests. Full
native BF16 generation is likely to need more VRAM than most free GPUs provide.

| Platform | Free GPU situation | Fit for this repo | Caveat |
| --- | --- | --- | --- |
| Google Colab Free | Free notebooks can access GPUs/TPUs, but resources are not guaranteed and usage limits fluctuate. | Good for scripted install/import or conversion smoke tests. | Colab free tier may terminate Web UI style usage; official FAQ says free runtimes are prioritized for interactive notebooks, not web services. |
| Kaggle Notebooks | Usually offers free notebook GPUs with quota. | Good for notebook-based import and possibly image-understanding smoke tests. | GPU type/quota must be checked at runtime; web UI exposure is awkward. |
| AWS SageMaker Studio Lab | Historically offers free notebook CPU/GPU sessions. | Good fallback for notebook smoke tests. | Availability and GPU session limits must be checked before use. |
| Lightning AI Studios | May provide starter/free credits depending on account/region. | Better than notebooks if it allows a web app port for ComfyUI. | Free-credit availability changes; verify before relying on it. |
| AutoDL / other hourly GPU clouds | Not free, but cheap and predictable. | Best practical fallback for full ComfyUI UI validation if Modal budget is gone. | Requires paying for enough VRAM; V100-32G may be tight but worth trying. |

Sources to verify before choosing a free platform:

- [Google Colab FAQ](https://research.google.com/colaboratory/faq.html)
- [Kaggle Notebooks documentation](https://www.kaggle.com/docs/notebooks)
- [AWS SageMaker Studio Lab FAQ](https://studiolab.sagemaker.aws/faq)
- [Lightning AI Studios](https://lightning.ai/studios)

## Related links

| Resource | Link |
| --- | --- |
| BAGEL paper | https://arxiv.org/abs/2505.14683 |
| BAGEL homepage | https://bagel-ai.org/ |
| Original BAGEL model | https://huggingface.co/ByteDance-Seed/BAGEL-7B-MoT |
| Recommended converted ComfyUI model | https://huggingface.co/6chan/bagel_comfy |
| BAGEL variant collection | https://huggingface.co/collections/6chan/bagel |
| Online demo | https://demo.bagel-ai.org/ |

## License

This project is licensed under the Apache 2.0 License. Please refer to the
official license terms for the use of the BAGEL model.
