#!/usr/bin/env python3
"""Download the small model pairs for the jspace testbed.

Pairs matter: safety alignment (RLHF/DPO) is the independent variable, so we need
base AND instruct checkpoints of the same families.

Usage: python src/download_models.py [smol|qwen|all]
"""
import sys
from huggingface_hub import snapshot_download

MODELS = {
    "smol": [
        "HuggingFaceTB/SmolLM2-135M-Instruct",
        "HuggingFaceTB/SmolLM2-135M",
        "HuggingFaceTB/SmolLM2-360M-Instruct",
        "HuggingFaceTB/SmolLM2-360M",
    ],
    "qwen": [
        "Qwen/Qwen2.5-0.5B-Instruct",
        "Qwen/Qwen2.5-0.5B",
        "Qwen/Qwen2.5-1.5B-Instruct",   # server-only (RAM)
        "Qwen/Qwen2.5-1.5B",            # base counterpart (P1 pair; added 2026-09-24)
    ],
}

def main(which: str):
    for key, names in MODELS.items():
        if which not in ("all", key):
            continue
        for name in names:
            print(f"== downloading {name}")
            snapshot_download(name, ignore_patterns=["*.gguf", "*.onnx*", "*consolidated*"])
            print(f"   done: {name}")

if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "all")
