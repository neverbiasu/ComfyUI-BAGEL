"""Immutable BAGEL capability and variant descriptors.

These dataclasses describe what a converted BAGEL model *structurally* supports.
They are derived from converted metadata plus the coupled-module keys present
in the safetensors header -- never from filename substrings. Runtime/backend
selection (DF11, RecA) is layered on top by :mod:`modeling.bagel.variants`.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import List


class CapabilityTier(str, Enum):
    """Evidence-based support tier. Documentation must never claim a higher tier."""

    NATIVE = "native"
    COMPATIBLE = "compatible"
    EXPERIMENTAL = "experimental"
    UNSUPPORTED = "unsupported"


@dataclass
class BagelCapabilities:
    text_to_image: bool = False
    image_edit: bool = False
    image_understanding: bool = False

    def to_list(self) -> List[str]:
        out = []
        if self.text_to_image:
            out.append("text_to_image")
        if self.image_edit:
            out.append("image_edit")
        if self.image_understanding:
            out.append("image_understanding")
        return out

    @classmethod
    def from_list(cls, items: List[str]) -> "BagelCapabilities":
        return cls(
            text_to_image="text_to_image" in items,
            image_edit="image_edit" in items,
            image_understanding="image_understanding" in items,
        )


@dataclass
class BagelVariantDescriptor:
    name: str = ""
    architecture: str = "Bagel"
    variant: str = ""
    dtype: str = ""
    quantization: str = "none"
    tier: CapabilityTier = CapabilityTier.EXPERIMENTAL
    capabilities: BagelCapabilities = field(default_factory=BagelCapabilities)
    detection_source: str = ""

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "architecture": self.architecture,
            "variant": self.variant,
            "dtype": self.dtype,
            "quantization": self.quantization,
            "tier": self.tier.value,
            "capabilities": self.capabilities.to_list(),
            "detection_source": self.detection_source,
        }


class VariantAdapter:
    """Protocol for structural variant adapters (e.g. DF11, RecA).

    An adapter declares which structural signature it handles and, when
    selected, builds the model-weight view for that variant. Importing the
    adapter must succeed without the optional backend (DFloat11) installed.
    """

    name: str = ""

    @staticmethod
    def detects(metadata, key_roots: set) -> bool:
        raise NotImplementedError

    def build(self, *args, **kwargs):
        raise NotImplementedError
