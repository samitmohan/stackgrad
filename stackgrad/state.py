import json

import numpy as np

_DTYPES = {"F32": np.float32, "F16": np.float16, "F64": np.float64, "I64": np.int64, "I32": np.int32, "U8": np.uint8}


def load_safetensors(path):
    return dict(iter_safetensors(path))


def iter_safetensors(path):
    with open(path, "rb") as f:
        n = int.from_bytes(f.read(8), "little")
        header = json.loads(f.read(n))
    raw = np.memmap(path, dtype=np.uint8, mode="r", offset=8 + n)
    for name, info in header.items():
        if name == "__metadata__":
            continue
        start, end = info["data_offsets"]
        b = raw[start:end]
        if info["dtype"] == "BF16":
            # bf16 is the top half of an f32, so shifting its bits up 16 rebuilds the f32 exactly
            arr = (b.view(np.uint16).astype(np.uint32) << 16).view(np.float32)
        else:
            arr = np.array(b.view(_DTYPES[info["dtype"]]))
        yield name, arr.reshape(info["shape"])
