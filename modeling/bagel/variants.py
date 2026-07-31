"""Structural BAGEL variant detection and registry.

Variants are detected from converted metadata plus the coupled-module key
roots present in the safetensors header -- never from filename substrings. The
known native variant is registered explicitly; any other detected structure is
reported as experimental with no executable capabilities until validated.
"""
from __future__ import annotations

from typing import Dict, Iterable, Set

from .converted_format import CANONICAL_MODULE_ROOTS, ConvertedBagelMetadata
from .model_types import (
    BagelCapabilities,
    BagelVariantDescriptor,
    CapabilityTier,
)

NATIVE_VARIANT = "BAGEL-7B-MoT"

# Capabilities granted to a fully validated native coupled BAGEL model.
_NATIVE_CAPABILITIES = BagelCapabilities(
    text_to_image=True, image_edit=True, image_understanding=True
)


def _key_roots(tensor_keys: Iterable[str]) -> Set[str]:
    return {k.split(".", 1)[0] for k in tensor_keys}


def detect_variant(
    metadata: ConvertedBagelMetadata, tensor_keys: Iterable[str]
) -> BagelVariantDescriptor:
    """Return a descriptor for the converted model based on evidence only."""
    roots = _key_roots(tensor_keys)
    missing_critical = [p for p in CANONICAL_MODULE_ROOTS if p not in roots]

    is_native = (
        metadata.variant == NATIVE_VARIANT
        and metadata.dtype == "bf16"
        and metadata.quantization in ("none", "")
        and not missing_critical
    )

    if is_native:
        return BagelVariantDescriptor(
            name=NATIVE_VARIANT,
            architecture=metadata.architecture,
            variant=metadata.variant,
            dtype=metadata.dtype,
            quantization=metadata.quantization,
            tier=CapabilityTier.NATIVE,
            capabilities=_NATIVE_CAPABILITIES,
            detection_source="metadata+keys",
        )

    if missing_critical:
        return BagelVariantDescriptor(
            name=metadata.variant or "unknown",
            architecture=metadata.architecture,
            variant=metadata.variant,
            dtype=metadata.dtype,
            quantization=metadata.quantization,
            tier=CapabilityTier.UNSUPPORTED,
            capabilities=BagelCapabilities(),
            detection_source=f"missing critical keys: {missing_critical}",
        )

    # Detected structure but not the validated native variant.
    return BagelVariantDescriptor(
        name=metadata.variant or "unknown",
        architecture=metadata.architecture,
        variant=metadata.variant,
        dtype=metadata.dtype,
        quantization=metadata.quantization,
        tier=CapabilityTier.EXPERIMENTAL,
        capabilities=BagelCapabilities(),
        detection_source="metadata+keys (unvalidated variant)",
    )


# Registry of named variant detectors for explicit lookup.
VARIANT_REGISTRY: Dict[str, BagelVariantDescriptor] = {}
