# ComfyUI-Native BAGEL Migration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Migrate ComfyUI-BAGEL to converted single-file BAGEL models, native ComfyUI latent workflows, official FLUX VAE nodes, and ComfyUI model lifecycle management without breaking legacy workflows.

**Architecture:** A converted safetensors file under `models/diffusion_models` contains the complete coupled BAGEL MoT, vision, connector, and latent-projection model. Native task nodes consume a patched `BAGEL_MODEL`, return or consume standard FLUX `LATENT` values, and leave VAE work to official nodes; legacy nodes and `models/bagel` remain available through explicit adapters.

**Tech Stack:** Python 3.10+, PyTorch, safetensors, ComfyUI backend APIs, Hugging Face tokenizers/transformers, pytest, Git, GitHub Fork branch `comfyui-native-migration`, AutoDL CUDA validation.

## Global Constraints

- Work only on branch `comfyui-native-migration`; do not use a `codex/` prefix.
- OpenCode implements one bounded task and stops with an uncommitted diff plus test evidence.
- Codex reviews scope, code, tests, and evidence. Commit only after Codex explicitly approves that task.
- Every approved task ends in exactly one focused commit using the command supplied by that task.
- Preserve current node class types, input names/defaults, output order/types, example workflow compatibility, and `models/bagel` discovery.
- New native nodes use converted `comfyui_bagel` safetensors under `models/diffusion_models` and official `VAELoader`, `VAEEncode`, and `VAEDecode` nodes.
- New `BAGEL Generate` and `BAGEL Edit` return standard `LATENT`; `BAGEL Understand` has no VAE dependency.
- The new loader does not create an inference pipeline or long-lived `InterleaveInferencer`.
- Tokenizer implementation and assets ship with this repository, not `models/text_encoders`.
- Do not automatically download models or install Python packages during node execution.
- Do not call ComfyUI global `unload_all_models()` from a loader.
- BAGEL-7B-MoT BF16 is the first required runtime target. DF11 and RecA follow through variant adapters.
- Before AutoDL access, state why remote CUDA is required, the commands to run, and the remote files that will change.

## Target File Map

- `__init__.py`: ComfyUI custom-node entrypoint and combined node mappings.
- `nodes.py`: existing legacy node definitions; preserve their public contracts.
- `nodes_model_loading.py`: converted BAGEL loader and loader-specific settings.
- `nodes_generation.py`: native generation, editing, and understanding nodes.
- `runtime.py`: VAE-free generation/editing/understanding runtime.
- `inferencer.py`: legacy image-returning inference path.
- `modeling/bagel/model_types.py`: immutable model, variant, capability, and runtime value types.
- `modeling/bagel/converted_format.py`: converted-format metadata and key validation.
- `modeling/bagel/model_loader.py`: converted-file discovery and complete BAGEL construction.
- `modeling/bagel/model_patcher.py`: ComfyUI lifecycle wrapper and reload factory.
- `modeling/bagel/variants.py`: structural registry plus BAGEL, DF11, and RecA adapters.
- `modeling/bagel/latent.py`: FLUX latent validation and BAGEL patchify/unpatchify.
- `modeling/qwen2/bagel_tokenizer.py`: tokenizer construction and fingerprint checks.
- `modeling/qwen2/tokenizer/`: source-packaged tokenizer assets.
- `scripts/convert_bagel_model.py`: offline raw-to-converted CLI.
- `tests/unit/`: pure Python, schema, tokenizer, latent, and meta-device tests.
- `tests/workflows/`: node-contract and workflow JSON regression tests.
- `tests/fixtures/`: small metadata and key-layout fixtures; never model weights.
- `example_workflows/`: existing legacy workflows plus native workflow JSON files.
- `docs/validation/`: local and AutoDL evidence.

Do not create a `comfyui_bagel/` directory. This repository root is already the custom-node Python package. Follow WanVideoWrapper's top-level `nodes_*.py` layout and keep architecture code in the existing `modeling/` tree. Root `tests/` follows Nunchaku; workflow JSON regression follows VideoHelperSuite.

---

### Task 1: Branch, Skill Installation, and Test Harness

**Files:**
- Create: `tests/conftest.py`
- Create: `tests/unit/test_node_registration.py`
- Install after approval: project-local ComfyUI custom-node skills under `.agents/skills/`

**Interfaces:**
- Consumes: existing `NODE_CLASS_MAPPINGS` and `NODE_DISPLAY_NAME_MAPPINGS` from `nodes.py`.
- Produces: a test harness that imports the repository root as a ComfyUI custom-node package and verifies the existing mappings unchanged.

- [ ] **Step 1: Create and switch to the implementation branch**

Run:

```bash
git switch -c comfyui-native-migration
```

Expected: `Switched to a new branch 'comfyui-native-migration'`.

- [ ] **Step 2: Install the reviewed custom-node skills project-locally**

Copy the nine `comfyui-node-*` skill directories from a reviewed clone of `jtydhr88/comfyui-custom-node-skills` into `.agents/skills/`. Do not install executable dependencies.

Run:

```bash
find .agents/skills -maxdepth 2 -name SKILL.md | sort
```

Expected: nine project-local ComfyUI skill entrypoints covering basics, inputs, outputs, datatypes, advanced, lifecycle, frontend, migration, and packaging.

- [ ] **Step 3: Write the failing import-preservation test**

```python
# tests/unit/test_node_registration.py
def test_root_entrypoint_preserves_all_legacy_nodes(custom_node_module):
    from nodes import NODE_CLASS_MAPPINGS as legacy

    classes = custom_node_module.NODE_CLASS_MAPPINGS
    displays = custom_node_module.NODE_DISPLAY_NAME_MAPPINGS
    assert set(legacy).issubset(classes)
    assert set(displays).issubset(classes)
```

- [ ] **Step 4: Run the test and verify the harness is absent**

Run: `pytest tests/unit/test_node_registration.py -v`

Expected: FAIL because the `custom_node_module` fixture is not defined.

- [ ] **Step 5: Add a ComfyUI-context import fixture**

```python
# tests/conftest.py
def install_fake_comfy_modules(monkeypatch):
    """Install only the folder_paths/comfy attributes required at import time."""
    # Add each explicit stub when the failing import identifies it. Runtime
    # behavior covered by an assertion must use a purpose-built fake instead.

@pytest.fixture
def custom_node_module(monkeypatch):
    install_fake_comfy_modules(monkeypatch)
    spec = importlib.util.spec_from_file_location(
        "ComfyUI_BAGEL", ROOT / "__init__.py",
        submodule_search_locations=[str(ROOT)],
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module
```

`install_fake_comfy_modules` supplies only the APIs needed during import. Do not create a second application package or change root mappings in this task.

- [ ] **Step 6: Run package and existing syntax checks**

Run:

```bash
pytest tests/unit/test_node_registration.py -v
python -m compileall -q nodes.py inferencer.py modeling tests
git diff --check
```

Expected: PASS, no compile output, no whitespace errors.

- [ ] **Step 7: Stop for Codex review**

Provide `git diff`, test output, and the list of installed skill files. Do not commit.

- [ ] **Step 8: Commit after Codex approval**

```bash
git add .agents/skills tests
git commit -m "test: add custom node test harness"
```

### Task 2: Converted Safetensors Schema and BF16 Converter

**Files:**
- Create: `modeling/bagel/converted_format.py`
- Create: `scripts/convert_bagel_model.py`
- Create: `tests/unit/test_converted_format.py`
- Create: `tests/unit/test_model_conversion.py`

**Interfaces:**
- Produces: `ConvertedBagelMetadata.from_safetensors(metadata: Mapping[str, str])`.
- Produces: `convert_checkpoint(source_dir: Path, output_file: Path, variant: str, dtype: str) -> ConversionManifest`.
- Output metadata includes `format=comfyui_bagel`, `format_version=1`, architecture, variant, source revision, dtype, tokenizer fingerprint, special-token IDs, latent format, capabilities, and converter version.

- [ ] **Step 1: Write schema rejection and acceptance tests**

```python
def test_schema_rejects_raw_checkpoint():
    with pytest.raises(ValueError, match="not a converted ComfyUI-BAGEL model"):
        ConvertedBagelMetadata.from_safetensors({})

def test_schema_accepts_v1_bagel():
    meta = ConvertedBagelMetadata.from_safetensors({
        "format": "comfyui_bagel",
        "format_version": "1",
        "architecture": "bagel_mot",
        "variant": "bagel-7b-mot",
        "dtype": "bfloat16",
        "tokenizer_fingerprint": "sha256:test",
        "latent_format": "flux",
        "capabilities": "generate,edit,understand",
    })
    assert meta.variant == "bagel-7b-mot"
```

- [ ] **Step 2: Run schema tests and confirm failure**

Run: `pytest tests/unit/test_converted_format.py -v`

Expected: FAIL because `ConvertedBagelMetadata` is undefined.

- [ ] **Step 3: Implement strict metadata parsing**

Use a frozen dataclass with an explicit `REQUIRED_KEYS` set. Reject unknown format versions, non-FLUX latent format, empty capabilities, and missing tokenizer fingerprints with messages asserted by tests.

- [ ] **Step 4: Write key-mapping identity tests**

Use representative keys for `language_model`, `vit_model`, `connector`, `vae2llm`, `llm2vae`, `latent_pos_embed`, and `time_embedder`. Assert that the converter strips only known source prefixes and rejects collisions.

- [ ] **Step 5: Implement deterministic key mapping and collision detection**

```python
def map_state_dict(source: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    converted = {}
    for source_key, tensor in source.items():
        target_key = normalize_key(source_key)
        if target_key in converted:
            raise ValueError(f"key collision: {source_key} -> {target_key}")
        converted[target_key] = tensor
    return converted
```

- [ ] **Step 6: Add a tiny-shard conversion CLI test**

Create temporary safetensors shards and an index JSON. Invoke `python scripts/convert_bagel_model.py` and assert one output file, v1 metadata, a JSON manifest, stable sorted keys, and matching SHA-256.

- [ ] **Step 7: Implement streaming conversion and manifest emission**

The CLI accepts `--source`, `--output`, `--variant`, and `--dtype`. It refuses to overwrite unless `--force` is passed, writes to a temporary sibling file, validates the completed artifact, then atomically renames it.

- [ ] **Step 8: Run conversion tests**

Run: `pytest tests/unit/test_converted_format.py tests/unit/test_model_conversion.py -v`

Expected: all PASS using only tiny synthetic tensors.

- [ ] **Step 9: Stop for Codex review, then commit after approval**

```bash
git add modeling/bagel/converted_format.py scripts/convert_bagel_model.py tests/unit
git commit -m "feat: add BAGEL safetensors converter"
```

### Task 3: Tokenizer, Variant Types, and Structural Registry

**Files:**
- Create: `modeling/bagel/model_types.py`
- Create: `modeling/bagel/variants.py`
- Create: `modeling/qwen2/bagel_tokenizer.py`
- Create: `modeling/qwen2/tokenizer/*`
- Create: `tests/unit/test_bagel_tokenizer.py`
- Create: `tests/unit/test_variant_registry.py`

**Interfaces:**
- Produces: `BagelCapabilities(generate: bool, edit: bool, understand: bool, multi_image_edit: bool)`.
- Produces: `BagelVariantDescriptor(id, architecture, dtype, quantization, capabilities)`.
- Produces: `load_bagel_tokenizer(expected_fingerprint, expected_special_ids)`.
- Produces: `VariantRegistry.detect(metadata, keys) -> BagelVariantAdapter`.

- [ ] **Step 1: Copy tokenizer assets from the immutable BAGEL source revision**

Record the source revision and SHA-256 hashes in `modeling/qwen2/tokenizer/MANIFEST.json`. Do not use mutable network downloads at runtime.

- [ ] **Step 2: Write tokenizer fingerprint and token-ID tests**

Assert deterministic vocabulary fingerprint and IDs for `<|im_start|>`, `<|im_end|>`, `<|vision_start|>`, and `<|vision_end|>`. Add a mismatch test expecting `ValueError: tokenizer fingerprint mismatch`.

- [ ] **Step 3: Implement repository-local tokenizer construction**

The loader resolves assets relative to `bagel.py`, creates `Qwen2Tokenizer`, registers only missing BAGEL special tokens, verifies IDs, and returns an immutable tokenizer bundle.

- [ ] **Step 4: Write structural registry tests**

Cover BAGEL BF16 detection, rejection of name-only spoofing, capability parsing, unknown format version, and an experimental descriptor that cannot execute generation.

- [ ] **Step 5: Implement the adapter protocol and BAGEL registry entry**

```python
class BagelVariantAdapter(Protocol):
    descriptor: BagelVariantDescriptor
    def validate_keys(self, keys: Collection[str]) -> None: ...
    def build_empty_model(self, config: Mapping[str, Any]) -> torch.nn.Module: ...
```

Detection must use converted metadata plus required structural keys, never repository-name substring checks.

- [ ] **Step 6: Run tests**

Run: `pytest tests/unit/test_bagel_tokenizer.py tests/unit/test_variant_registry.py -v`

Expected: all PASS.

- [ ] **Step 7: Stop for review, then commit after approval**

```bash
git add modeling/bagel/model_types.py modeling/bagel/variants.py modeling/qwen2/bagel_tokenizer.py modeling/qwen2/tokenizer tests/unit
git commit -m "feat: add BAGEL tokenizer and variant registry"
```

### Task 4: ComfyUI Patcher, Converted Model Loader, and Native Loader Node

**Files:**
- Create: `modeling/bagel/model_patcher.py`
- Create: `modeling/bagel/model_loader.py`
- Create: `nodes_model_loading.py`
- Modify: `__init__.py`
- Create: `tests/unit/test_model_loader.py`
- Create: `tests/unit/test_model_patcher.py`
- Create: `tests/workflows/test_loader_node.py`

**Interfaces:**
- Produces: `BagelModelHandle(model, patcher, tokenizer, config, variant, checkpoint_path, cache_key)`.
- Produces: `load_converted_bagel(path: Path, model_options: Mapping[str, Any]) -> BagelModelHandle`.
- Produces node class type `BagelNativeModelLoader`, returning `("BAGEL_MODEL",)`.

- [ ] **Step 1: Write discovery tests**

Mock `folder_paths.get_filename_list("diffusion_models")`. Assert the node shows only files whose headers declare `format=comfyui_bagel`, and that a raw model produces a conversion-specific error.

- [ ] **Step 2: Write lifecycle tests with fake ComfyUI modules**

Assert load/offload devices come from `comfy.model_management`, no call reaches `unload_all_models`, the patcher exposes a disk-backed reload factory, and repeated node execution reuses ComfyUI caching rather than rereading the file.

- [ ] **Step 3: Implement `BagelModelPatcher` and reload factory**

Wrap the complete model, not its vision and language submodules separately. Keep DF11 hooks out of the BF16 implementation.

- [ ] **Step 4: Implement converted model loading**

Read metadata before tensor allocation, detect the adapter, construct on meta device, load the complete coupled state dict, verify no critical missing/unexpected keys, create tokenizer bundle, and return `BagelModelHandle`.

- [ ] **Step 5: Implement the native loader node contract**

Inputs: converted model name, weight dtype, load-device choice, and advanced attention option. No auto-download field. Register under `BAGEL/Native` while leaving all old mappings unchanged.

- [ ] **Step 6: Run tests**

Run:

```bash
pytest tests/unit/test_model_loader.py tests/unit/test_model_patcher.py tests/workflows/test_loader_node.py -v
python -m compileall -q nodes_model_loading.py modeling
```

Expected: all PASS and no compile output.

- [ ] **Step 7: Stop for review, then commit after approval**

```bash
git add modeling/bagel/model_patcher.py modeling/bagel/model_loader.py nodes_model_loading.py __init__.py tests
git commit -m "feat: load converted BAGEL models natively"
```

### Task 5: VAE-Free Latent Runtime

**Files:**
- Create: `modeling/bagel/latent.py`
- Create: `runtime.py`
- Create: `tests/unit/test_latent.py`
- Create: `tests/unit/test_runtime.py`

**Interfaces:**
- Produces: `patchify_flux_latent(samples: Tensor, patch_size: int) -> Tensor`.
- Produces: `unpatchify_bagel_latent(tokens: Tensor, height: int, width: int, patch_size: int) -> Tensor`.
- Produces: `BagelRuntime.generate(...) -> dict[str, Any]` with `samples` shaped `[B, 16, H/8, W/8]`.
- Produces: `BagelRuntime.edit(image, encoded_latent, ...) -> dict[str, Any]`.
- Produces: `BagelRuntime.understand(image, prompt, ...) -> str`.

- [ ] **Step 1: Write latent round-trip tests**

Use non-square latent tensors, verify exact round trips, reject channels other than 16, and reject spatial dimensions incompatible with patch size.

- [ ] **Step 2: Implement FLUX latent validation and transforms**

Return tensors without applying extra scale or shift; official `VAEEncode` already returns FLUX-format encoded values and `VAEDecode` expects the same format.

- [ ] **Step 3: Write runtime contract tests**

Use fake model methods to assert generation never calls VAE encode/decode, editing uses both original image and supplied latent, understanding does not access VAE state, and generation returns `{"samples": tensor}`.

- [ ] **Step 4: Extract a VAE-free runtime from `InterleaveInferencer`**

Move KV-context, text, vision, sampling, and latent unpacking into `BagelRuntime`. Keep the old inferencer untouched until the legacy task. Do not return PIL images from native methods.

- [ ] **Step 5: Run runtime tests**

Run: `pytest tests/unit/test_latent.py tests/unit/test_runtime.py -v`

Expected: all PASS.

- [ ] **Step 6: Stop for review, then commit after approval**

```bash
git add modeling/bagel/latent.py runtime.py tests/unit
git commit -m "feat: add VAE-free BAGEL latent runtime"
```

### Task 6: Native Generate, Edit, and Understand Nodes

**Files:**
- Create: `nodes_generation.py`
- Modify: `__init__.py`
- Create: `tests/workflows/test_generation_node.py`
- Create: `tests/workflows/test_editing_node.py`
- Create: `tests/workflows/test_understanding_node.py`
- Create: `tests/workflows/test_native_workflows.py`
- Create: `example_workflows/bagel_native_text_to_image.json`
- Create: `example_workflows/bagel_native_image_edit.json`
- Create: `example_workflows/bagel_native_image_understanding.json`

**Interfaces:**
- Produces `BagelNativeGenerate`: `BAGEL_MODEL -> (LATENT, STRING)`.
- Produces `BagelNativeEdit`: `BAGEL_MODEL + IMAGE + LATENT -> (LATENT, STRING)`.
- Produces `BagelNativeUnderstand`: `BAGEL_MODEL + IMAGE + STRING -> (STRING,)`.

- [ ] **Step 1: Write exact node-contract tests**

Assert input names, defaults, return types, categories, and that execution errors propagate instead of returning black images or fake success.

- [ ] **Step 2: Implement minimal node wrappers**

Nodes validate public inputs, seed RNG through one shared helper, call `BagelRuntime`, and return ComfyUI tuples. No node imports or accepts a VAE object.

- [ ] **Step 3: Write workflow-graph tests**

Parse each JSON and assert:

- text-to-image connects native generation `LATENT` to official `VAEDecode`;
- editing connects official `VAEEncode` plus original `IMAGE` to native edit, then native `LATENT` to official `VAEDecode`;
- understanding has no `VAELoader`, `VAEEncode`, or `VAEDecode` node.

- [ ] **Step 4: Add minimal native workflow JSON fixtures**

Use stable node class types and no machine-specific absolute paths.

- [ ] **Step 5: Run node and workflow tests**

Run: `pytest tests/workflows/test_generation_node.py tests/workflows/test_editing_node.py tests/workflows/test_understanding_node.py tests/workflows/test_native_workflows.py -v`

Expected: all PASS.

- [ ] **Step 6: Stop for review, then commit after approval**

```bash
git add nodes_generation.py __init__.py tests/workflows example_workflows
git commit -m "feat: add native BAGEL workflow nodes"
```

### Task 7: Legacy Compatibility Layer and Regression Tests

**Files:**
- Modify: `nodes.py`
- Modify: `inferencer.py`
- Modify: `__init__.py`
- Create: `tests/workflows/test_legacy_node_contracts.py`
- Create: `tests/workflows/test_legacy_model_discovery.py`
- Create: `tests/workflows/test_legacy_workflows.py`

**Interfaces:**
- Consumes: all current public legacy class types and the native internal services where behavior matches.
- Produces: unchanged legacy loader/task contracts and one once-per-session migration warning.

- [ ] **Step 1: Snapshot all existing node contracts in tests**

Record class type, required/optional fields, defaults, return types, return names, functions, and categories for every current node. Parse every existing example workflow and assert all referenced class types remain registered.

- [ ] **Step 2: Write legacy discovery tests**

Assert `models/bagel/<repo-layout>` remains discoverable and the native `diffusion_models` scan does not hide or move legacy folders.

- [ ] **Step 3: Implement compatibility facades**

Keep old image-returning behavior, including internal VAE decode, because changing it would corrupt saved workflow links. Delegate shared validation and sampling helpers where semantics are identical.

- [ ] **Step 4: Add once-per-session deprecation logging**

Use a module-level guarded logger call. Do not emit warnings per denoising step or per image.

- [ ] **Step 5: Run full CPU regression suite**

Run:

```bash
pytest tests -v
python -m compileall -q .
git diff --check
```

Expected: all tests PASS, no compile errors, no whitespace errors.

- [ ] **Step 6: Stop for review, then commit after approval**

```bash
git add nodes.py inferencer.py __init__.py tests/workflows
git commit -m "refactor: preserve legacy BAGEL workflows"
```

### Task 8: DF11, RecA, and Capability Matrix

**Files:**
- Modify: `modeling/bagel/variants.py`
- Create: `docs/compatibility.md`
- Create: `tests/unit/test_df11_adapter.py`
- Create: `tests/unit/test_reca_adapter.py`
- Create: `tests/unit/test_capability_matrix.py`

**Interfaces:**
- Produces structural adapters with the same protocol defined in Task 3.
- Produces a generated compatibility table whose labels are `native`, `compatible`, `experimental`, or `unsupported`.

- [ ] **Step 1: Add structural fixtures from immutable model metadata**

Fixtures contain headers, configs, and key names only; do not commit large weights.

- [ ] **Step 2: Write DF11 adapter tests**

Assert dedicated backend requirement, prohibited secondary quantization, correct capabilities, and a clear missing-backend error.

- [ ] **Step 3: Implement DF11 adapter behind optional import boundaries**

Importing the custom node must succeed when DFloat11 is absent. Only selecting a DF11 file may require the backend.

- [ ] **Step 4: Write and implement RecA structural tests**

Validate its source revision, required key differences, tokenizer fingerprint, and supported capabilities before labeling it compatible.

- [ ] **Step 5: Generate the capability matrix from registry data**

The documentation test fails if a registered adapter is absent from `docs/compatibility.md` or if documentation claims a higher tier than registry evidence.

- [ ] **Step 6: Run adapter tests**

Run: `pytest tests/unit/test_df11_adapter.py tests/unit/test_reca_adapter.py tests/unit/test_capability_matrix.py -v`

Expected: all PASS without requiring GPU or optional DF11 installation.

- [ ] **Step 7: Stop for review, then commit after approval**

```bash
git add modeling/bagel/variants.py docs/compatibility.md tests/unit
git commit -m "feat: add BAGEL variant adapters"
```

### Task 9: Packaging, Documentation, and Local Release Checks

**Files:**
- Modify: `README.md`
- Modify: `pyproject.toml`
- Modify: `requirements.txt`
- Create: `docs/model-conversion.md`
- Create: `docs/native-workflows.md`
- Create: `docs/legacy-migration.md`
- Create: `tests/test_packaging.py`

**Interfaces:**
- Produces installable package metadata and complete operator instructions for converted models, native workflows, and legacy behavior.

- [ ] **Step 1: Write packaging tests**

Assert tokenizer assets are included in source/wheel packaging, no runtime installer is registered, package dependencies match requirements, and all documented workflow paths exist.

- [ ] **Step 2: Update packaging metadata**

Include tokenizer assets explicitly. Keep optional quantization backends optional and document their extras instead of importing them at module load.

- [ ] **Step 3: Write conversion and native workflow documentation**

Document `models/diffusion_models`, `models/vae`, retained `models/bagel`, converter commands, hashes/manifests, official VAE graph wiring, and error meanings.

- [ ] **Step 4: Write legacy migration documentation**

Show old and new node chains, state that old nodes and directories remain supported, and give no removal date.

- [ ] **Step 5: Run local release checks**

Run:

```bash
pytest tests -v
python -m build
python -m compileall -q nodes.py nodes_model_loading.py nodes_generation.py inferencer.py runtime.py modeling
git diff --check
git status --short
```

Expected: tests PASS, sdist/wheel build succeeds, compile and diff checks are silent, status contains only reviewed Task 9 changes and build artifacts ignored by Git.

- [ ] **Step 6: Stop for review, then commit after approval**

```bash
git add README.md pyproject.toml requirements.txt docs tests/test_packaging.py
git commit -m "docs: document native BAGEL workflows"
```

### Task 10: Push Fork Branch and Complete AutoDL Acceptance

**Files:**
- Create: `docs/validation/autodl-environment.md`
- Create: `docs/validation/autodl-results.md`
- Create: `docs/validation/workflows/*.json`
- Create: `docs/validation/logs/`
- Create: `docs/validation/outputs/` for small representative artifacts only

**Interfaces:**
- Consumes: reviewed commits from Tasks 1-9 and converted BAGEL BF16 plus FLUX AE.
- Produces: reproducible remote evidence for install, registration, native workflows, legacy regression, hashes, peak VRAM, and load/offload behavior.

- [ ] **Step 1: Verify branch and clean status before push**

Run:

```bash
git branch --show-current
git status --short
git log --oneline --decorate -10
```

Expected: branch is `comfyui-native-migration`, status is clean, and each reviewed task has one commit.

- [ ] **Step 2: Push the Fork branch**

Run: `git push -u origin comfyui-native-migration`

Expected: branch created or updated on the user's GitHub Fork.

- [ ] **Step 3: Announce and record remote scope before AutoDL access**

Record GPU model, ComfyUI revision, Python/PyTorch/CUDA versions, remote checkout path, models to mount/download, and commands to execute. State that remote writes are limited to the selected ComfyUI custom-node checkout, model paths, workflow files, and validation outputs.

- [ ] **Step 4: Install from the pushed branch**

Clone or update the custom node from the Fork branch, install declared requirements in the ComfyUI Python environment, and start ComfyUI with logs captured.

Expected: no `IMPORT FAILED` entry for ComfyUI-BAGEL.

- [ ] **Step 5: Verify node registration**

Query `/object_info` and assert native and legacy class types are present with expected input/output contracts.

- [ ] **Step 6: Run native text-to-image**

Use converted BAGEL BF16, official FLUX `VAELoader`, native generation, and official `VAEDecode`. Save workflow, output, seed, timings, peak VRAM, checkpoint SHA-256, and relevant log excerpt.

- [ ] **Step 7: Run native image editing**

Use official `VAEEncode`, original image, native edit, and official `VAEDecode`. Save the same evidence fields.

- [ ] **Step 8: Run native image understanding**

Verify no VAE node is present and save prompt, response, timing, peak VRAM, and logs.

- [ ] **Step 9: Run one legacy workflow regression**

Use `models/bagel` and an existing example workflow. Confirm unchanged output types and successful execution.

- [ ] **Step 10: Verify model lifecycle**

Repeat an unchanged prompt and confirm no repeated checkpoint disk load. Switch or unload workflows and capture evidence that ComfyUI model management can offload/reload the BAGEL patcher.

- [ ] **Step 11: Write evidence summaries and run final audit**

Run locally after syncing evidence:

```bash
pytest tests -v
git diff --check
git status --short
```

Expected: all tests PASS and only validation evidence is uncommitted.

- [ ] **Step 12: Stop for Codex review, then commit and push after approval**

```bash
git add docs/validation
git commit -m "test: record AutoDL BAGEL validation"
git push origin comfyui-native-migration
```

## Final Completion Checklist

- [ ] Every task has one reviewed commit.
- [ ] `comfyui-native-migration` exists on the GitHub Fork.
- [ ] Converted BAGEL BF16 installs from `models/diffusion_models`.
- [ ] Official FLUX VAE nodes handle native encoding and decoding.
- [ ] Generate and Edit return standard `LATENT`.
- [ ] Understand runs without a VAE.
- [ ] Legacy nodes and `models/bagel` still work.
- [ ] AutoDL generation, editing, understanding, and legacy regression evidence is committed.
- [ ] DF11 and RecA documentation matches actual validation status.
