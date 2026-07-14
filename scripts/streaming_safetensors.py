"""Dependency-free streaming safetensors writer.

This module deliberately imports nothing beyond the standard library so it can
be reused and unit-tested without ``torch`` or ``safetensors`` installed. It
emits a valid ``.safetensors`` file one tensor at a time: only one tensor's raw
bytes are held by the caller, while the data section is spooled to a sidecar
temp file. After all tensors are written, :meth:`StreamingSafetensorsWriter.finalize`
emits the 8-byte length prefix, the JSON header (including embedded metadata
under the ``__metadata__`` block), and streams the spooled data into the final
file.

The safetensors format requires tensor ``data_offsets`` to be contiguous -- no
gaps may be inserted between payloads. The JSON header is padded with spaces to
an 8-byte boundary so the data section begins at an 8-byte aligned file offset;
the padded length is what is encoded in the 8-byte prefix. Peak memory is
bounded by the largest single tensor plus the transient raw and (in the caller)
casted copies held while one tensor is being added -- never the whole checkpoint.

Format reference: https://huggingface.co/docs/safetensors
"""
from __future__ import annotations

import json
import os
import struct
from typing import List, Optional, Tuple

# Valid safetensors dtype codes accepted by add(). These match the codes
# emitted by safetensors.torch (see its `_TYPES` map): F8_E4M3 (not F8_E4M3FN)
# and F8_E5M2 are the two float8 codes. The generic writer accepts the full set
# of standard codes so it can be reused for any safetensors payload; the
# converter restricts itself to the dtypes it actually emits (BF16/F16/F32).
SAFETENSORS_DTYPES = {
    "BF16",
    "F16",
    "F32",
    "F64",
    "F8_E4M3",
    "F8_E5M2",
    "I8",
    "I16",
    "I32",
    "I64",
    "U8",
    "BOOL",
}

# JSON header is padded so the data section begins at an 8-byte file offset.
_ALIGN = 8


class StreamingSafetensorsWriter:
    """Write a safetensors file incrementally with bounded memory.

    Only one tensor's raw bytes are held by the caller at a time; tensor
    payloads are written contiguously (no gaps) to a sidecar temp file. After
    all tensors are written, :meth:`finalize` emits the 8-byte length prefix,
    the JSON header (including embedded metadata), and streams the spooled data
    into the final file.
    """

    def __init__(self, data_path: str) -> None:
        self._data_path = data_path
        self._f = open(data_path, "wb")
        # (name, dtype_code, shape, data_begin, data_end) -- offsets within the
        # spooled data section (0-based, contiguous).
        self._entries: List[Tuple[str, str, List[int], int, int]] = []
        self._names = set()
        self._offset = 0

    def add(self, name: str, dtype_code: str, shape: List[int], raw_bytes: bytes) -> None:
        if name in self._names:
            raise ValueError(f"duplicate tensor name: {name!r}")
        if dtype_code not in SAFETENSORS_DTYPES:
            raise ValueError(f"unsupported safetensors dtype code: {dtype_code!r}")
        begin = self._offset
        self._f.write(raw_bytes)
        end = self._offset + len(raw_bytes)
        self._offset = end
        self._entries.append((name, dtype_code, shape, begin, end))
        self._names.add(name)

    def close(self) -> None:
        if not self._f.closed:
            self._f.close()

    def finalize(self, out_path: str, metadata_json: Optional[str] = None) -> None:
        header: dict = {}
        if metadata_json is not None:
            header["__metadata__"] = {"comfyui_bagel": metadata_json}
        for name, dtype_code, shape, begin, end in self._entries:
            header[name] = {
                "dtype": dtype_code,
                "shape": shape,
                "data_offsets": [begin, end],
            }
        header_json = json.dumps(header, separators=(",", ":")).encode("utf-8")
        pad = (_ALIGN - (len(header_json) % _ALIGN)) % _ALIGN
        header_json = header_json + b" " * pad
        with open(out_path, "wb") as out:
            out.write(struct.pack("<Q", len(header_json)))
            out.write(header_json)
            with open(self._data_path, "rb") as data:
                while True:
                    chunk = data.read(1 << 20)
                    if not chunk:
                        break
                    out.write(chunk)


def cleanup_temp(data_path: str) -> None:
    if os.path.exists(data_path):
        os.remove(data_path)
