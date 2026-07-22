"""Converted single-file BAGEL safetensors schema and key normalization.

Defines the metadata header carried in every converted ``comfyui_bagel`` file,
the deterministic state-dict key normalization, and the conversion manifest
structure. The BAGEL coupled checkpoint already uses canonical module-prefixed
state-dict keys (``language_model.*``, ``vit_model.*``, ``connector.*``,
``vae2llm.*``, ``llm2vae.*``, ``latent_pos_embed.*``, ``vit_pos_embed.*``,
``time_embedder.*``); normalization is the single place that preserves them,
rejects collisions, and asserts the critical coupled weights are present.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Dict, List, Optional

FORMAT_NAME = "comfyui_bagel"
FORMAT_VERSION = 1
CONVERTER_VERSION = "1.0.0"

# Canonical coupled-module roots that must survive conversion.
CANONICAL_MODULE_ROOTS = (
    "language_model",
    "vit_model",
    "connector",
    "vae2llm",
    "llm2vae",
    "latent_pos_embed",
    "vit_pos_embed",
    "time_embedder",
)

# Top-level prefixes whose absence makes a converted file unusable.
CRITICAL_PREFIXES = tuple(CANONICAL_MODULE_ROOTS)

DTYPE_TO_TORCH = {
    "bf16": "bfloat16",
    "fp16": "float16",
    "fp32": "float32",
}


@dataclass
class ConvertedBagelMetadata:
    """Metadata embedded in the converted safetensors ``__metadata__`` header."""

    format: str = FORMAT_NAME
    format_version: int = FORMAT_VERSION
    architecture: str = "Bagel"
    variant: str = ""
    source_repository: str = ""
    source_revision: str = ""
    source_hashes: Dict[str, str] = field(default_factory=dict)
    dtype: str = "bf16"
    quantization: str = "none"
    tokenizer_fingerprint: str = ""
    special_token_ids: Dict[str, int] = field(default_factory=dict)
    latent_format: str = "flux"
    capabilities: List[str] = field(default_factory=list)
    tensor_summary: Dict[str, object] = field(default_factory=dict)
    # Embedded model configs so the loader never downloads them at runtime.
    model_configs: Dict[str, object] = field(default_factory=dict)
    # Variant-specific Bagel constructor facts. These are descriptive until a
    # matching runtime adapter explicitly consumes them.
    model_options: Dict[str, object] = field(default_factory=dict)
    # Tokens appended by a variant on top of the packaged BAGEL tokenizer.
    # Keeping them in metadata lets a future adapter reproduce the exact
    # tokenizer contract without guessing from the filename.
    additional_special_tokens: List[str] = field(default_factory=list)
    converter_version: str = CONVERTER_VERSION

    def to_dict(self) -> Dict:
        d = {
            "format": self.format,
            "format_version": self.format_version,
            "architecture": self.architecture,
            "variant": self.variant,
            "source_repository": self.source_repository,
            "source_revision": self.source_revision,
            "source_hashes": dict(self.source_hashes),
            "dtype": self.dtype,
            "quantization": self.quantization,
            "tokenizer_fingerprint": self.tokenizer_fingerprint,
            "special_token_ids": dict(self.special_token_ids),
            "latent_format": self.latent_format,
            "capabilities": list(self.capabilities),
            "tensor_summary": dict(self.tensor_summary),
            "model_configs": dict(self.model_configs),
            "model_options": dict(self.model_options),
            "additional_special_tokens": list(self.additional_special_tokens),
            "converter_version": self.converter_version,
        }
        return d

    @classmethod
    def from_dict(cls, d: Dict) -> "ConvertedBagelMetadata":
        return cls(
            format=d.get("format", FORMAT_NAME),
            format_version=int(d.get("format_version", FORMAT_VERSION)),
            architecture=d.get("architecture", "Bagel"),
            variant=d.get("variant", ""),
            source_repository=d.get("source_repository", ""),
            source_revision=d.get("source_revision", ""),
            source_hashes=dict(d.get("source_hashes", {})),
            dtype=d.get("dtype", "bf16"),
            quantization=d.get("quantization", "none"),
            tokenizer_fingerprint=d.get("tokenizer_fingerprint", ""),
            special_token_ids=dict(d.get("special_token_ids", {})),
            latent_format=d.get("latent_format", "flux"),
            capabilities=list(d.get("capabilities", [])),
            tensor_summary=dict(d.get("tensor_summary", {})),
            model_configs=dict(d.get("model_configs", {})),
            model_options=dict(d.get("model_options", {})),
            additional_special_tokens=list(d.get("additional_special_tokens", [])),
            converter_version=d.get("converter_version", CONVERTER_VERSION),
        )

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True)

    @classmethod
    def from_json(cls, s: str) -> "ConvertedBagelMetadata":
        return cls.from_dict(json.loads(s))


@dataclass
class ConversionManifest:
    """Sidecar describing the conversion inputs, outputs, and key mapping."""

    source: str = ""
    source_hashes: Dict[str, str] = field(default_factory=dict)
    converted: str = ""
    converted_hash: str = ""
    variant: str = ""
    dtype: str = "bf16"
    key_mapping: Dict[str, str] = field(default_factory=dict)
    # Self-describing tensor shape/dtype record: {name: [shape, dtype_str]}.
    tensor_details: Dict[str, object] = field(default_factory=dict)
    converter_version: str = CONVERTER_VERSION

    def to_json(self) -> str:
        return json.dumps(self.__dict__, sort_keys=True, indent=2)

    @classmethod
    def from_json(cls, s: str) -> "ConversionManifest":
        return cls(**json.loads(s))


def normalize_key(raw_key: str) -> str:
    """Deterministic conversion of a raw state-dict key to its converted form.

    The BAGEL coupled checkpoint already uses canonical module prefixes, so the
    identity mapping is correct today. This function is the single, auditable
    place to adjust if a future source variant ships a different key scheme.
    """
    return raw_key


def normalize_state_dict(state_dict: Dict[str, "object"]) -> Dict[str, "object"]:
    """Return ``{normalized_key: tensor}`` and the ``{raw: normalized}`` mapping.

    Rejects collisions (two raw keys mapping to the same normalized key with
    differing values) and asserts the critical coupled-module prefixes exist.
    """
    out: Dict[str, object] = {}
    mapping: Dict[str, str] = {}
    for raw_key, tensor in state_dict.items():
        norm = normalize_key(raw_key)
        if norm in out:
            raise KeyError(f"key collision after normalization: {norm!r} (from {raw_key!r})")
        out[norm] = tensor
        mapping[raw_key] = norm

    present_roots = {k.split(".", 1)[0] for k in out}
    missing = [p for p in CRITICAL_PREFIXES if p not in present_roots]
    if missing:
        raise KeyError(f"converted state dict missing critical module weights: {missing}")
    return out, mapping


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()
