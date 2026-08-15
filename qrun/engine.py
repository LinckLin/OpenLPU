"""qrun engine assembly: build a RunEngine for a given qbin + model + dtype."""
from __future__ import annotations

from compiler.isa.qbin import read_qbin
from qforge.safetensors import SafeTensors
from qrun import program as P
from qrun import weights as W
from qrun.qmetal import QMetal, HbmPlan
from qrun.runtime import RunEngine

H = 1024
VOCAB = 151936
BLOCK = 128


def build_engine(qbin_path: str, model_dir: str, dtype: str, *, tokenizer=None,
                 ref=None, slab_shift: int = 22,
                 calibration_prompt: str | None = None,
                 awq: bool = False,
                 weights_from_hf: bool = False) -> RunEngine:
    """Load qbin + model, build QMetal, programs, and inject gamma.

    INT8 activation-scale calibration uses the golden projection inputs of
    `calibration_prompt` (default: a short representative sentence); pass an
    explicit prompt to calibrate on a specific evaluation input.

    INT4 (`awq=False`) quantizes weights plain symmetric per-128-group;
    `awq=True` searches AWQ-style per-(n,g) scales from the calibration
    activations of `calibration_prompt` (qforge.quant.quantize_weight_int4_awq).

    BF16 loads the 141 projection weights from the qbin tensors table by
    default (the BF16 container); `weights_from_hf=True` selects the legacy
    safetensors path instead. The container's default-dtype flag must be BF16
    (an INT8 container is rejected with an error, no silent fallback).
    """
    qbin = read_qbin(qbin_path)
    st = SafeTensors.open(f"{model_dir}/model.safetensors")
    qmetal = QMetal(slab_shift)

    if dtype == "int8":
        act_scales = None
        if ref is not None and tokenizer is not None:
            calib = calibration_prompt or \
                "The transformer architecture processes sequences in parallel."
            act_scales = W.calibrate_act_scales(ref, tokenizer, calib)
        layouts = W.int8_layouts(qbin, qmetal, act_scales)
        weights_end = W.int8_weights_end(qbin)
    elif dtype == "int4":
        act_inputs = None
        if awq and ref is not None and tokenizer is not None:
            calib = calibration_prompt or \
                "The transformer architecture processes sequences in parallel."
            act_inputs = W.calibrate_act_inputs(ref, tokenizer, calib)
        layouts, weights_end = W.int4_layouts(st, qmetal, act_inputs=act_inputs)
    elif dtype == "bf16":
        if weights_from_hf:
            layouts, weights_end = W.bf16_layouts(st, qmetal)
        else:
            W.check_bf16_container(qbin)
            layouts, weights_end = W.bf16_layouts_from_qbin(qbin, qmetal)
    else:
        raise ValueError(f"unknown dtype {dtype!r}")

    input_bytes = BLOCK * H * 2              # 256 KiB
    logits_bytes = BLOCK * VOCAB * 2         # 38.9 MB (PF lm_head output)
    plan = qmetal.plan_hbm(weights_end, input_bytes=input_bytes,
                           logits_bytes=logits_bytes)

    W.inject_gamma(st, qmetal)
    W.inject_act_scale(qmetal, P._layout(1), P._layout(BLOCK))

    engine = RunEngine(qmetal, plan, dtype, layouts, st, tokenizer, ref)
    engine.build_programs()
    return engine
