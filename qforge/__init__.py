"""qforge — QCore HF -> .qbin compile frontend (P4, M3).

Pipeline: config parse -> build graph -> safetensors load -> W8A8 quantize
(symmetric per-128-K-group, weight side; activation scale is a compile-time
constant) -> weight repack -> tiling (N<=128, K streamed) -> scheduling ->
00-container .qbin (magic/version/flags/header/tensors/pf_program/dc_program/
ENDQ + length check).

Reuses (read-only) compiler/isa/isa.py + compiler/isa/qbin.py + compiler/
lowering.py memory-plan conventions, and qsim/executor.py for functional
verification. Never writes outside qforge/ and docs/p4/.
"""

__version__ = "0.1.0"
__all__ = ["config", "graph", "safetensors", "quant", "lowering", "build",
           "cli"]
