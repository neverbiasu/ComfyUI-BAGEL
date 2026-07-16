# Deprecated BAGEL installation path

This page keeps the old all-in-one BAGEL installation path for users who still
need the deprecated `Bagel*` nodes. New users should prefer the native
single-file model layout described in the main README.

## Deprecated model layout

| Model | Old location | Deprecated loader |
| --- | --- | --- |
| `ByteDance-Seed/BAGEL-7B-MoT` | `ComfyUI/models/bagel/BAGEL-7B-MoT` | `BAGEL Model Loader (Deprecated)` |
| `DFloat11/BAGEL-7B-MoT-DF11` | `ComfyUI/models/bagel/BAGEL-7B-MoT-DF11` | `BAGEL Model Loader (Deprecated)` |

## Old manual download commands

### Standard model

```bash
git lfs install
git clone https://huggingface.co/ByteDance-Seed/BAGEL-7B-MoT ComfyUI/models/bagel/BAGEL-7B-MoT
```

Or:

```bash
python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='ByteDance-Seed/BAGEL-7B-MoT', local_dir='ComfyUI/models/bagel/BAGEL-7B-MoT')"
```

### DFloat11 quantized model

```bash
git clone https://huggingface.co/DFloat11/BAGEL-7B-MoT-DF11 ComfyUI/models/bagel/BAGEL-7B-MoT-DF11
```

Or:

```bash
python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='DFloat11/BAGEL-7B-MoT-DF11', local_dir='ComfyUI/models/bagel/BAGEL-7B-MoT-DF11')"
```

## Migration options

| If you have... | Do this |
| --- | --- |
| Existing original BAGEL BF16 shards | Convert them with `scripts/convert_bagel_model.py`. |
| No local BAGEL model yet | Download the converted single-file model from `6chan/bagel_comfy`. |
| Existing old workflows | Use the `_deprecated` workflows as a bridge, then migrate to the native workflows. |

## Deprecated dependencies

| Feature | Extra dependency |
| --- | --- |
| NF4 / INT8 legacy quantization | `bitsandbytes` |
| DFloat11 legacy model | `dfloat11` |
| Text display in old workflows | `comfyui-custom-scripts` |
