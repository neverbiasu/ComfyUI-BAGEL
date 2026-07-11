# ComfyUI-Native BAGEL Migration Design

**Date:** 2026-07-11  
**Status:** Approved design  
**Implementation branch:** `comfyui-native-migration`

## 1. Objective

Refactor ComfyUI-BAGEL into a ComfyUI-native custom-node package while preserving existing workflows. The new path loads a converted single-file BAGEL model from `models/diffusion_models`, exchanges standard ComfyUI `LATENT` values with task nodes, and delegates VAE loading, encoding, and decoding to ComfyUI's official nodes.

The migration must also support a controlled model-family expansion based on the BAGEL variants tracked in `/Users/nev4rb14su/workspace/research/msc-project/docs/related_work`. The first supported set is BAGEL-7B-MoT, BAGEL DF11, and BAGEL-RecA. Other related-work variants enter a capability matrix before they can be advertised as supported.

## 2. Design Principles

1. Follow actual model boundaries rather than creating one node per conceptual component.
2. Keep the complete BAGEL MoT, vision branch, tokenizer, connectors, and latent projections in one main model object.
3. Use official `VAELoader`, `VAEEncode`, and `VAEDecode` nodes for the FLUX AE path.
4. Return standard ComfyUI `LATENT` objects from generation and editing nodes.
5. Use ComfyUI model-management and patcher lifecycles instead of treating Accelerate dispatch hooks as the permanent runtime architecture.
6. Preserve existing node class types, fields, outputs, workflows, and `models/bagel` discovery during the migration.
7. Require converted ComfyUI safetensors for the new loader. Do not make the new loader interpret raw Hugging Face shard layouts.
8. Keep unsupported variants and capabilities explicit.
9. Implement in small reviewed tasks. Every task is committed only after review passes.

## 3. Reference Implementations

Implementation should use these repositories as concrete references:

- ComfyUI core: loader registration, `LATENT`, FLUX latent format, VAE nodes, model management, patcher reload factories, and tokenizer packaging.
- `kijai/ComfyUI-WanVideoWrapper`: custom main-model loader, component boundaries, model patching, dtype and offload settings, and direct node composition without an Assemble node.
- `kijai/ComfyUI-KJNodes`: model-loader conventions and node packaging patterns.
- `Kosinkadink/ComfyUI-VideoHelperSuite`: package structure, workflow-based tests, documentation, and compatibility discipline.
- `nunchaku-ai/ComfyUI-nunchaku`: specialized `ModelPatcher` behavior, quantized model handling, native ComfyUI interoperability, tests, and packaging.
- `jtydhr88/comfyui-custom-node-skills`: V3-first node, lifecycle, datatype, migration, packaging, and frontend guidance. Install this after the written spec is approved and before implementation begins.
- `Kijai/WanVideo_comfy`: converted single-file model distribution.
- `Comfy-Org/Omnigen2_ComfyUI_repackaged`: `split_files` packaging aligned to ComfyUI model directories.

## 4. Architecture

The target implementation has four layers:

```text
ComfyUI nodes
  - BAGEL Model Loader
  - BAGEL Generate
  - BAGEL Edit
  - BAGEL Understand
  - legacy compatibility nodes
        |
Runtime
  - conditioning and KV-context construction
  - BAGEL latent patchify/unpatchify
  - sampling and capability checks
        |
Model
  - complete BAGEL MoT and vision model
  - tokenizer and special-token protocol
  - variant adapter
  - ComfyUI model patcher
        |
Official ComfyUI nodes
  - VAELoader
  - VAEEncode
  - VAEDecode
```

There is no visible Assemble node. The complete BAGEL object must be constructed at model-load time because its checkpoint contains coupled `language_model`, `vit_model`, `connector`, `vae2llm`, `llm2vae`, position-embedding, and timestep modules. VAE weights are not part of the `Bagel` module and remain external.

The new model loader must not create an `InterleaveInferencer`. Task nodes construct only the lightweight runtime state needed for an execution.

## 5. Native Workflows and Node Contracts

### 5.1 Text-to-image

```text
BAGEL Model Loader -> BAGEL Generate -> LATENT -> VAEDecode -> IMAGE
                                               ^
VAELoader -------------------------------------|
```

`BAGEL Generate` inputs:

- `BAGEL_MODEL`
- prompt
- seed
- image dimensions or aspect-ratio preset
- sampling and CFG parameters

Outputs:

- standard `LATENT`
- optional reasoning/planning text

The node must not accept or load a VAE.

### 5.2 Image editing

```text
Load Image -------------------------------> BAGEL Edit
    |                                           |
    +-> VAEEncode -> LATENT --------------------+
                                                |
                                                v
                                             LATENT -> VAEDecode -> IMAGE
                                                          ^
VAELoader ------------------------------------------------+
```

`BAGEL Edit` receives both:

- the original `IMAGE` for the SigLIP/NaViT path;
- the official `VAEEncode` `LATENT` for BAGEL VAE-token conditioning.

It returns a standard `LATENT`. BAGEL-specific patchification stays inside the BAGEL runtime because it represents MoT token layout, not VAE behavior.

### 5.3 Image understanding

`BAGEL Understand` receives `BAGEL_MODEL`, `IMAGE`, and a prompt, and returns `STRING`. It does not load or accept a VAE.

### 5.4 Multi-image editing

Multi-image editing remains a separate capability-gated node. It uses the same native latent interfaces and rejects unsupported variants before expensive execution begins.

## 6. BAGEL Model Object

The loader returns a named `BAGEL_MODEL` object rather than the current loose dictionary. It contains:

- complete `Bagel` model;
- ComfyUI-compatible patcher;
- BAGEL tokenizer;
- special-token IDs;
- immutable model configuration;
- transforms required by the vision and interleaved-conditioning paths;
- variant descriptor;
- source identity and converted-format version;
- capability flags;
- runtime cache identity.

The object does not contain VAE weights or a long-lived inferencer.

## 7. Model Conversion and Distribution

### 7.1 New format

Raw BAGEL Hugging Face shards are converted offline into one ComfyUI safetensors file per main-model variant. The complete coupled BAGEL weights remain in that file.

Example distribution repository:

```text
BAGEL_ComfyUI_repackaged/
└── split_files/
    ├── diffusion_models/
    │   ├── bagel_7b_mot_bf16.safetensors
    │   ├── bagel_7b_mot_df11.safetensors
    │   └── bagel_reca_bf16.safetensors
    └── vae/
        └── ae.safetensors
```

Installed paths:

```text
ComfyUI/models/diffusion_models/*.safetensors
ComfyUI/models/vae/ae.safetensors
```

`diffusion_models` is appropriate because the complete BAGEL model is the main latent-generating flow model. The directory does not imply a traditional UNet architecture.

### 7.2 Tokenizer packaging

Do not place tokenizer assets in `models/text_encoders`. That directory is for text-encoder weights. BAGEL's tokenizer implementation and assets ship with the custom node, following ComfyUI's source-packaged tokenizer pattern:

```text
comfyui_bagel/tokenizers/qwen2_bagel/
```

The loader chooses the tokenizer implementation from model metadata and validates its vocabulary hash and special-token IDs. No tokenizer node is exposed.

### 7.3 Legacy format

`models/bagel/<original-repository-layout>/` remains supported by legacy nodes. New and legacy model layouts may coexist. The migration must not move or delete user model files.

The new loader lists only converted files under `diffusion_models` whose metadata identifies the ComfyUI-BAGEL format. If it is directed to a raw layout, it returns a conversion instruction instead of attempting a partial load.

### 7.4 Required metadata

Each converted file records at least:

- `format = comfyui_bagel`;
- format version;
- architecture and variant;
- source repository and immutable source revision;
- source shard hashes;
- dtype and quantization;
- tokenizer family and vocabulary hash;
- special-token IDs;
- `latent_format = flux`;
- supported capability flags;
- conversion tool version.

The converter emits a manifest with source hashes, destination hash, key mapping and counts, tensor shapes and dtypes, and all missing or unexpected keys. Missing critical weights fail conversion.

## 8. Variant Registry

Variant support is driven by structural adapters rather than repository-name substring checks. Each adapter defines:

- config and checkpoint-key detection;
- construction procedure;
- supported dtypes and quantization modes;
- tokenizer protocol;
- capability flags;
- required and forbidden keys;
- known limitations;
- validation status.

Initial support tiers:

| Variant | Initial tier | Requirement |
|---|---|---|
| BAGEL-7B-MoT | Native | BF16 conversion, native generation/edit/understanding smoke |
| BAGEL DF11 | Native adapter | Dedicated DF11 load and memory path |
| BAGEL-RecA | Compatible | Structural and behavioral validation against its source implementation |
| Other related-work variants | Detected/Experimental | Detection and capability reporting only until validated |

Quantization support is enabled only after separate validation. A converted filename or repository label is not proof of compatibility.

## 9. Model Management

The model layer follows ComfyUI lifecycle semantics:

- use ComfyUI load and offload devices;
- wrap the main model in a BAGEL-specific patcher compatible with ComfyUI model management;
- register a disk-backed reload factory for cache invalidation, cloning, and future multi-GPU support;
- do not call global `unload_all_models()` from a loader;
- avoid repeated checkpoint reads when unchanged nodes execute again;
- keep initial support at whole-model load/offload;
- treat block swapping as a later capability, not an MVP requirement.

DF11 may require a specialized adapter, but it must still expose the same lifecycle contract to task nodes.

## 10. Legacy Compatibility

The migration preserves current workflow JSON behavior:

- retain existing node class types;
- retain input field names and defaults;
- retain output order and types;
- retain `models/bagel` scanning;
- retain legacy image-returning task nodes;
- register old nodes under a Legacy category without removing them;
- emit a once-per-session deprecation notice;
- document the native replacement workflow.

Legacy nodes may use a facade over new internal services where semantics match. Where native and legacy contracts differ, especially internal VAE decoding versus external latent output, compatibility code must preserve old behavior explicitly rather than silently changing outputs.

No removal date is set in the first migration release.

## 11. Errors and Observability

New nodes raise contextual errors rather than returning black images or fake-success strings. Errors distinguish:

- unconverted/raw checkpoint;
- unknown format version;
- unsupported variant or capability;
- missing or unexpected weights;
- tokenizer mismatch;
- unavailable quantization backend;
- dtype/device incompatibility;
- out-of-memory conditions.

Logs include the detected variant, converted-format version, selected dtype, patcher load/offload devices, and checkpoint identity without exposing sensitive environment values.

## 12. Verification Strategy

### 12.1 Unit tests

- conversion key mapping and metadata;
- variant detection;
- tokenizer hashes and special-token IDs;
- latent patchify/unpatchify round trips;
- capability gates;
- legacy adapter contracts.

### 12.2 CPU/meta-device integration

- safetensors header and manifest validation;
- meta-device construction;
- node registration and contracts;
- converted-file filtering under `diffusion_models`;
- legacy example workflow parsing;
- absence of import-time optional-backend failures.

### 12.3 GPU smoke tests

- BAGEL text-to-image with official `VAEDecode`;
- BAGEL editing with official `VAEEncode` and `VAEDecode`;
- BAGEL image understanding without VAE loading;
- fixed-seed converted-versus-legacy comparison;
- repeat execution without repeated checkpoint reads;
- model unload/reload behavior;
- peak VRAM capture.

### 12.4 Acceptance gates

- the model loader creates no pipeline or inferencer;
- generation and editing return standard FLUX `LATENT` values;
- official FLUX AE nodes encode/decode correctly;
- understanding does not depend on a VAE;
- old example workflows still open and execute;
- converted BAGEL BF16 is behaviorally equivalent to the legacy BF16 path within a defined numeric or image tolerance;
- unsupported variants remain clearly labeled;
- conversion and runtime evidence is stored in the repository.

## 13. GitHub Fork and AutoDL Acceptance

Implementation happens on the GitHub Fork branch:

```text
comfyui-native-migration
```

The branch is pushed before remote acceptance. AutoDL is required because CUDA execution, realistic VRAM behavior, DF11, and ComfyUI model lifecycle cannot be validated on the local machine.

Before remote access, the operator states why AutoDL is needed, what commands will run, and what remote files will change.

AutoDL acceptance procedure:

1. Use a clean or explicitly selected ComfyUI environment.
2. Install the custom node from the Fork branch.
3. Install declared dependencies without a hidden runtime installer.
4. Download or mount converted BAGEL and FLUX AE files.
5. Start ComfyUI and confirm there are no import failures.
6. Confirm node registration through `/object_info`.
7. Build and save native text-to-image, image-editing, and image-understanding workflows.
8. Execute all three workflows with fixed inputs.
9. Save workflow JSON, logs, outputs, model hashes, peak VRAM, and load/offload evidence.
10. Run at least one legacy workflow regression.
11. Validate BAGEL BF16 first. Validate DF11 and RecA when their converted weights are available.
12. Commit reviewed acceptance evidence to the branch.

Release requires a pushed Fork branch, clean installation from that branch, three passing native workflows, one passing legacy workflow, and matching converted-file hashes. Unavailable experimental variants do not block release when they are labeled accurately.

## 14. Repository Structure

```text
comfyui_bagel/
├── nodes/
│   ├── loaders.py
│   ├── generation.py
│   ├── editing.py
│   ├── understanding.py
│   └── legacy.py
├── models/
│   ├── bagel_model.py
│   ├── patcher.py
│   └── variants/
├── runtime/
│   ├── inference.py
│   ├── conditioning.py
│   └── latent.py
├── conversion/
│   ├── convert.py
│   ├── key_mapping.py
│   └── manifest.py
└── tokenizers/
    └── qwen2_bagel/
```

Existing upstream-derived model code may be migrated incrementally rather than moved wholesale in the first task. File moves must serve a reviewed task and preserve import compatibility.

## 15. OpenCode Implementation and Review Loop

Codex owns the specification, review, and acceptance gates. OpenCode implements one bounded task at a time.

For every task:

```text
Codex task specification
-> OpenCode implementation and evidence, without commit
-> Codex diff, test, and scope review
-> OpenCode correction when required
-> Codex approval
-> one task-specific commit
-> task and evidence status update
```

OpenCode must not commit before review. Review checks scope, legacy compatibility, ComfyUI lifecycle use, duplicate model loading, error behavior, tests, and whether the diff forms one coherent commit.

Planned task order:

1. Install the ComfyUI custom-node skills and add package/test scaffolding without behavior changes.
2. Specify and implement the converted safetensors schema and BF16 converter.
3. Add conversion equivalence and manifest validation.
4. Implement the converted BAGEL model object, variant registry, tokenizer, patcher, and loader.
5. Refactor inference to accept and return native latents without owning a VAE.
6. Add native text-to-image generation.
7. Add native image editing with image and pre-encoded latent inputs.
8. Add native image understanding without VAE dependency.
9. Add and test legacy compatibility adapters.
10. Add DF11 and RecA adapters and the broader detection matrix.
11. Add documentation, converted-model instructions, and native example workflows.
12. Push the Fork branch and complete reviewed AutoDL acceptance.

Each task ends in a commit only after Codex approves the implementation and evidence.

## 16. Out of Scope for the Initial Migration

- upstreaming BAGEL support into ComfyUI core;
- a separate vision-loader node;
- an Assemble node;
- exposing a tokenizer node;
- pretending BAGEL is a standard `CLIP` implementation;
- block swapping before the native BF16 path works;
- automatic dependency installation during node execution;
- automatic model downloads in the core loader;
- claiming support for every indexed BAGEL derivative before validation;
- removing legacy nodes or `models/bagel` support.

