# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""VMEM floorplan, roofline bounds, and verdict tests for TPU advisor."""

from __future__ import annotations

import ast
import dataclasses
import math
from pathlib import Path

import pytest

from tests.megakernel.core.megakernel_fakes import TINY_TILES, make_fake_model_spec
from tests.megakernel.test_advisor_specs import DEEPSEEK_V3_CFG, QWEN35_397B_CFG
from vllm_torchtpu.megakernel import advisor
from vllm_torchtpu.megakernel.advisor import constants, spec_bridge, support
from vllm_torchtpu.megakernel.advisor.roofline import (
    _build_scope_context,
    _compute_scope_matmuls,
    _core_time,
    _evaluate_scopes,
    _resolve_calibrated_priors,
    _routed_chain,
)
from vllm_torchtpu.megakernel.advisor.verdict import (
    _apply_box,
    _assumed_box,
    _evaluate_corners,
)
from vllm_torchtpu.megakernel.advisor.vmem import _mesh_divisibility_reasons
from vllm_torchtpu.megakernel.core.errors import MegakernelConfigError
from vllm_torchtpu.megakernel.core.exec_spec import ExecSpec, Tiles

QWEN35_08B_CFG = {
    "model_type": "qwen3_5",
    "num_hidden_layers": 24,
    "hidden_size": 1024,
    "intermediate_size": 3584,
    "num_attention_heads": 8,
    "num_key_value_heads": 2,
    "head_dim": 256,
    "full_attention_interval": 4,
    "linear_num_key_heads": 16,
    "linear_num_value_heads": 16,
    "linear_key_head_dim": 128,
    "linear_value_head_dim": 128,
    "linear_conv_kernel_dim": 4,
    "top_k": 50,
    "quantization_config": {"quant_method": "fp8"},
}
MIXTRAL_8X7B_CFG = {
    "model_type": "mixtral",
    "num_hidden_layers": 32,
    "hidden_size": 4096,
    "intermediate_size": 14336,
    "num_attention_heads": 32,
    "num_key_value_heads": 8,
    "num_local_experts": 8,
    "num_experts_per_tok": 2,
    "torch_dtype": "bfloat16",
}


def _preset_workload(
    layout: advisor.ParallelLayout, **changes: object
) -> advisor.WorkloadSpec:
    """Return a WorkloadSpec on a deployment preset layout with `changes` applied."""
    mesh = advisor.WorkloadSpec(
        tp=layout.tp,
        ep=layout.ep,
        dp=layout.dp,
        pp=layout.pp,
        dcp=layout.dcp,
        pcp=layout.pcp,
        layer_groups=layout.layer_groups,
    )
    return dataclasses.replace(mesh, **changes)


def test_per_token_vmem_live_interval_and_sharding() -> None:
    """Verify per-token live VMEM interval math and TP/EP sharding on Kimi-K3.1-Pro."""
    pk3 = advisor.resolve_model_profile("kimi-k3.1-pro")
    wl_grp = advisor.WorkloadSpec(
        phase="verify",
        batch=16,
        spec_tokens=3,
        tp=32,
        ep=32,
        moe_mode="grouped_on_chip",
    )
    parts_grp = advisor.compute_per_token_vmem_bytes(pk3, wl_grp)
    expected_grp_tok = parts_grp["residual_buffer_bytes"] + max(
        parts_grp["attn_scratch_bytes"],
        parts_grp["linear_attn_scratch_bytes"],
        parts_grp["moe_scratch_bytes"],
    )
    assert parts_grp["total_per_token_bytes"] == expected_grp_tok
    assert parts_grp["b_tok"] == expected_grp_tok
    rep_grp = advisor.advise(pk3, wl_grp, "tpu7x")
    cross_sc = next(s for s in rep_grp.scopes if s.scope == "whole_step")
    one_l_sc = next(s for s in rep_grp.scopes if s.scope == "single_layer")
    min_verify_rows = 64  # Minimum verify tokens for Qwen 3.5
    assert cross_sc.max_vmem_tokens >= min_verify_rows
    assert one_l_sc.max_vmem_tokens > cross_sc.max_vmem_tokens

    wl_msk = dataclasses.replace(wl_grp, moe_mode="masked_on_chip")
    parts_msk = advisor.compute_per_token_vmem_bytes(pk3, wl_msk)
    assert parts_grp["total_per_token_bytes"] < parts_msk["total_per_token_bytes"]

    wl_no_alias = dataclasses.replace(wl_grp, vmem_alias=False)
    parts_no_alias = advisor.compute_per_token_vmem_bytes(pk3, wl_no_alias)
    assert parts_no_alias["total_per_token_bytes"] > parts_grp["total_per_token_bytes"]


def test_multi_tpu_vmem_and_fp8_dtype_physics() -> None:
    """Verify TPU FP8 physics, VMEM buffer overflow, Mixtral MoE, and DeepSeek MLA."""
    c7x = advisor.resolve_chip_spec("tpu7x")
    cv6e = advisor.resolve_chip_spec("v6e")

    tpu7x_fp8_flops = 2.30e15  # TPU v7x FP8 peak FLOP/s
    tpu7x_fp8_row_floor = 32  # TPU v7x FP8 row floor
    v6e_fp8_flops = 9.20e14  # TPU v6e FP8 peak FLOP/s

    assert (c7x.native_ocp_fp8, c7x.peak_flops("fp8"), c7x.lhs_row_floor("fp8")) == (
        True,
        tpu7x_fp8_flops,
        tpu7x_fp8_row_floor,
    )
    assert (cv6e.native_ocp_fp8, cv6e.peak_flops("fp8")) == (
        False,
        v6e_fp8_flops,
    )
    assert (
        c7x.vmem_usable_bytes,
        cv6e.vmem_usable_bytes,
    ) == (62 << 20, 126 << 20)

    p397 = advisor.model_profile_from_hf_config(QWEN35_397B_CFG, name="qwen35-397b")
    rep_v6e_tp8 = advisor.advise(p397, advisor.WorkloadSpec(tp=8, batch=4), cv6e)
    assert rep_v6e_tp8.hbm_fits is False

    pmix = advisor.model_profile_from_hf_config(MIXTRAL_8X7B_CFG, name="mixtral-8x7b")
    assert pmix.num_moe_layers == MIXTRAL_8X7B_CFG["num_hidden_layers"]
    assert pmix.moe_intermediate_size == MIXTRAL_8X7B_CFG["intermediate_size"]
    ffn_dim_test = 2048
    p_ffn = advisor.model_profile_from_hf_config(
        {
            "model_type": "mixtral",
            "num_hidden_layers": 2,
            "hidden_size": 512,
            "ffn_dim": ffn_dim_test,
            "num_attention_heads": 8,
            "num_local_experts": 4,
            "num_experts_per_tok": 2,
        }
    )
    assert p_ffn.intermediate_size == ffn_dim_test
    assert p_ffn.moe_intermediate_size == ffn_dim_test


def test_deepseek_v3_mla_kv_split_physics() -> None:
    """Verify DeepSeek-V3 MLA parameter sizing and kv_split HBM savings."""
    c7x = advisor.resolve_chip_spec("tpu7x")
    pdsv3 = advisor.model_profile_from_hf_config(DEEPSEEK_V3_CFG, name="deepseek-v3")
    assert pdsv3.head_dim == DEEPSEEK_V3_CFG["qk_nope_head_dim"]
    assert (
        pdsv3.shared_expert_intermediate_size
        == DEEPSEEK_V3_CFG["moe_intermediate_size"]
    )
    assert pdsv3.num_moe_layers == (
        DEEPSEEK_V3_CFG["num_hidden_layers"] - DEEPSEEK_V3_CFG["first_k_dense_replace"]
    )
    assert pdsv3.is_periodic is False
    pdsv3_dense_only = dataclasses.replace(
        pdsv3, num_moe_layers=0, num_experts=0, top_k=0, intermediate_size=0
    )
    r_mla_only = advisor.advise(
        pdsv3_dense_only, advisor.WorkloadSpec(tp=8, batch=1, context=4096), c7x
    )
    nh_loc = pdsv3_dense_only.num_attention_heads // 8
    q_tot = nh_loc * (
        pdsv3_dense_only.qk_nope_head_dim + pdsv3_dense_only.qk_rope_head_dim
    )
    q_dim = nh_loc * pdsv3_dense_only.head_dim
    mla_layer_params = (
        pdsv3_dense_only.q_lora_rank * (pdsv3_dense_only.hidden_size + q_tot)
        + pdsv3_dense_only.hidden_size
        * (pdsv3_dense_only.kv_lora_rank + pdsv3_dense_only.qk_rope_head_dim + q_dim)
        + pdsv3_dense_only.kv_lora_rank
        * nh_loc
        * (pdsv3_dense_only.qk_nope_head_dim + pdsv3_dense_only.head_dim)
        + 3 * pdsv3_dense_only.hidden_size
    )
    expected_mla_w_gb = round((pdsv3_dense_only.num_layers * mla_layer_params) / 1e9, 4)
    assert r_mla_only.scopes[0].w_step_gb == expected_mla_w_gb
    r_mla_pre = advisor.advise(
        pdsv3_dense_only,
        advisor.WorkloadSpec(phase="prefill", tp=8, batch=1, isl=4096, chunk_size=1),
        c7x,
    )
    assert r_mla_only.step_arithmetic_intensity > r_mla_pre.step_arithmetic_intensity
    r_ds1 = advisor.advise(
        pdsv3, advisor.WorkloadSpec(tp=8, batch=32, context=8192), c7x
    )
    r_ds2 = advisor.advise(
        pdsv3, advisor.WorkloadSpec(tp=8, kv_split=2, batch=32, context=8192), c7x
    )
    assert r_ds2.hbm_used_gb < r_ds1.hbm_used_gb
    assert [
        (s.feasible, s.support, "unmodeled_layer_family" in s.blockers)
        for s in r_ds1.scopes
        if s.scope == "whole_step"
    ] == [(True, "hypothetical", True)]
    verify_batch, verify_spec = 8, 3
    mixed_chunk = 128
    r_ver = advisor.advise(
        pdsv3,
        advisor.WorkloadSpec(
            phase="verify", tp=8, batch=verify_batch, spec_tokens=verify_spec
        ),
        c7x,
    )
    r_mix = advisor.advise(
        pdsv3,
        advisor.WorkloadSpec(
            phase="mixed",
            tp=8,
            batch=verify_batch,
            spec_tokens=verify_spec,
            chunk_size=mixed_chunk,
        ),
        c7x,
    )
    assert r_ver.active_tokens == verify_batch * (verify_spec + 1)
    assert r_mix.active_tokens == verify_batch * (verify_spec + 1) + mixed_chunk


def test_asymptotic_decode_intensity_and_vmem_fit_headroom() -> None:
    """Verify `asymptotic_decode_intensity`, `fit_vmem_budget_bytes`, `w_crit_gb`."""
    assert advisor.fit_vmem_budget_bytes() == 62 << 20
    assert advisor.fit_vmem_budget_bytes(Tiles(vmem_mib=62)) == 62 << 20
    p08 = advisor.model_profile_from_hf_config(
        QWEN35_08B_CFG, name="qwen35-0.8b", state_dtype="fp32"
    )
    r_b16 = advisor.advise(
        p08,
        advisor.WorkloadSpec(phase="verify", batch=16, spec_tokens=3, tp=8),
        "tpu7x",
    )
    r_b128 = advisor.advise(
        p08,
        advisor.WorkloadSpec(phase="verify", batch=128, spec_tokens=3, tp=8),
        "tpu7x",
    )
    assert r_b16.asymptotic_decode_intensity is not None
    assert r_b16.asymptotic_decode_intensity == r_b128.asymptotic_decode_intensity
    assert r_b16.asymptotic_decode_intensity < r_b16.ridge_dense_flop_per_byte
    assert r_b16.step_arithmetic_intensity < r_b16.asymptotic_decode_intensity

    r_alias = advisor.advise(
        p08, advisor.WorkloadSpec(phase="prefill", batch=1, isl=2048, tp=8), "tpu7x"
    )
    r_no_alias = advisor.advise(
        p08,
        advisor.WorkloadSpec(
            phase="prefill",
            batch=1,
            isl=2048,
            tp=8,
            vmem_alias=False,
            vmem_fit_headroom=True,
        ),
        "tpu7x",
    )
    r_no_alias_disp = advisor.advise(
        p08,
        advisor.WorkloadSpec(
            phase="prefill",
            batch=1,
            isl=2048,
            tp=8,
            vmem_alias=False,
            vmem_fit_headroom=True,
            op_overhead_us=5.0,
        ),
        "tpu7x",
    )
    # 64 MiB physical minus 2 MiB reserve
    expected_vmem_mib = float(constants.DEFAULT_USABLE_VMEM_MIB)
    assert r_alias.vmem_budget_mib == expected_vmem_mib
    assert r_no_alias.vmem_budget_mib == advisor.fit_vmem_budget_bytes() / (1 << 20)
    cx_alias = next(s for s in r_alias.scopes if s.scope == "whole_step")
    cx_no_al = next(s for s in r_no_alias.scopes if s.scope == "whole_step")
    cx_disp = next(s for s in r_no_alias_disp.scopes if s.scope == "whole_step")
    assert cx_alias.n_splits == 1
    assert cx_no_al.n_splits > 1
    assert cx_no_al.w_crit_gb is not None
    assert cx_disp.w_crit_gb is not None
    assert cx_disp.w_crit_gb > cx_no_al.w_crit_gb
    assert cx_disp.step_bound_us > cx_no_al.step_bound_us


def test_all_four_fusion_scopes_reachable() -> None:
    """Prove `whole_step`, `single_layer`, `stage`, and `per_op` each win."""
    p08 = advisor.model_profile_from_hf_config(QWEN35_08B_CFG, name="qwen35-0.8b")
    p397 = advisor.model_profile_from_hf_config(QWEN35_397B_CFG, name="qwen35-397b")
    pllama = advisor.resolve_model_profile("llama-3.1-405b")

    # 1. `whole_step` (`whole_step`) wins on 397B TP8 decode and 0.8B prefill.
    r_dec = advisor.advise(
        p397, advisor.WorkloadSpec(phase="decode", batch=16, tp=8), "tpu7x"
    )
    assert r_dec.recommended_scope == "whole_step"
    assert r_dec.recommended_moe_mode == "masked_on_chip"
    assert r_dec.build_megakernel is True
    r_b32 = advisor.advise(
        p397, advisor.WorkloadSpec(phase="decode", batch=32, tp=8), "tpu7x"
    )
    assert r_b32.recommended_scope == "whole_step"
    assert r_b32.recommended_moe_mode == "grouped_on_chip"
    r_08_pre = advisor.advise(
        p08, advisor.WorkloadSpec(phase="prefill", batch=1, isl=2048, tp=8), "tpu7x"
    )
    assert r_08_pre.fastest_scope == "whole_step"
    assert r_08_pre.recommended_moe_mode == "dense_mlp"

    # 2. `single_layer` (`layer`) wins in the B=488 VMEM cliff band on tpu7x,
    # whereas v6e (TP16) and v5p Megacore (TP8) keep `whole_step` in 1 pass.
    r_cliff_7x = advisor.advise(
        p397,
        advisor.WorkloadSpec(phase="decode", batch=488, context=1024, tp=8),
        "tpu7x",
    )
    assert r_cliff_7x.fastest_scope == "single_layer"
    assert r_cliff_7x.recommended_moe_mode == "grouped_on_chip"
    r_cliff_v6e = advisor.advise(
        p397,
        advisor.WorkloadSpec(phase="decode", batch=488, context=1024, tp=16),
        "v6e",
    )
    assert r_cliff_v6e.fastest_scope == "whole_step"
    r_ep8_sat = advisor.advise(
        p397,
        advisor.WorkloadSpec(
            phase="decode", batch=600, tp=2, dp=4, ep=8, tiles=Tiles()
        ),
        "tpu7x",
    )
    # Saturated DP4 x TP2 decode with locked wide ring: 150 tok/core overflow
    # the wide whole-step ceiling while single_layer fits in 1 pass.
    assert r_ep8_sat.fastest_scope == "single_layer"
    assert r_ep8_sat.recommended_moe_mode == "grouped_on_chip"

    r_llama_pre = advisor.advise(
        pllama, advisor.WorkloadSpec(phase="prefill", batch=1, isl=4096, tp=8), "tpu7x"
    )
    assert r_llama_pre.recommended_scope == "per_op"
    assert r_llama_pre.recommended_moe_mode == "dense_mlp"
    assert r_llama_pre.build_megakernel is False

    # 3. `layer_group` wins on PP8 tail and segmented TP8 (`segment_layers=10`).
    r_stg = advisor.advise(
        p397,
        advisor.WorkloadSpec(
            phase="prefill", batch=1, isl=4096, chunk_size=64, tp=1, pp=8
        ),
        "tpu7x",
    )
    assert r_stg.fastest_scope == "layer_group"
    r_seg = advisor.advise(
        p397,
        advisor.WorkloadSpec(phase="decode", batch=16, tp=8, segment_layers=10),
        "tpu7x",
    )
    assert r_seg.fastest_scope == "layer_group"
    # 6 segments of 10 layers across 3 ops per segment
    expected_ops_per_step = (p397.num_layers // 10) * 3
    assert (
        next(s for s in r_seg.scopes if s.scope == "layer_group").ops_per_step
        == expected_ops_per_step
    )
    r_16k = advisor.advise(
        p397,
        advisor.WorkloadSpec(phase="prefill", batch=8, isl=16384, tp=1, pp=8),
        "tpu7x",
    )
    assert r_16k.recommended_scope == "per_op"
    assert r_16k.build_megakernel is False


def test_vmem_budget_override_and_dma_buffers_triple_buffering() -> None:
    """Verify user-configurable `--vmem-budget-mib` and `--dma-buffers {2,3}`."""
    p397 = advisor.resolve_model_profile("qwen3.5-397b")
    rep_2buf = advisor.advise(
        p397, advisor.WorkloadSpec(tp=8, ep=8, batch=16, dma_buffers=2), "tpu7x"
    )
    rep_3buf = advisor.advise(
        p397, advisor.WorkloadSpec(tp=8, ep=8, batch=16, dma_buffers=3), "tpu7x"
    )
    rep_custom_vmem = advisor.advise(
        p397,
        advisor.WorkloadSpec(tp=8, ep=8, batch=16, dma_buffers=2, vmem_budget_mib=58.0),
        "tpu7x",
    )
    cx_2 = next(s for s in rep_2buf.scopes if s.scope == "whole_step")
    cx_3 = next(s for s in rep_3buf.scopes if s.scope == "whole_step")
    cx_vmem = next(s for s in rep_custom_vmem.scopes if s.scope == "whole_step")
    # 64 MiB physical minus 2 MiB reserve
    expected_vmem_mib = float(constants.DEFAULT_USABLE_VMEM_MIB)
    custom_vmem_mib = 58.0  # Custom VMEM budget in MiB
    assert rep_2buf.vmem_budget_mib == expected_vmem_mib
    assert rep_custom_vmem.vmem_budget_mib == custom_vmem_mib
    assert cx_vmem.max_vmem_tokens < cx_2.max_vmem_tokens
    assert cx_3.vmem_static_mib > cx_2.vmem_static_mib
    assert cx_3.max_vmem_tokens < cx_2.max_vmem_tokens
    assert cx_3.step_bound_us >= cx_2.step_bound_us


@pytest.mark.parametrize(
    ("tp", "ep", "pcp", "accepted"),
    [(4, 8, 2, True), (4, 8, 1, False), (4, 16, 2, False)],
)
def test_expert_parallel_spans_context_parallel_ranks(
    tp: int, ep: int, pcp: int, *, accepted: bool
) -> None:
    """Verify `ep` may span `tp * dp * pcp` ranks and no wider."""
    p397 = advisor.resolve_model_profile("qwen3.5-397b")
    wl = advisor.WorkloadSpec(phase="prefill", batch=1, isl=4096, tp=tp, ep=ep, pcp=pcp)
    if accepted:
        rep = advisor.advise(p397, wl, "tpu7x")
        assert rep.active_tokens == 4096 // pcp
    else:
        with pytest.raises(MegakernelConfigError):
            advisor.advise(p397, wl, "tpu7x")


def test_advise_from_model_spec_and_planner() -> None:
    """Verify `advise_from_spec` bridges periodic and non-periodic `ModelSpec`."""
    ms = make_fake_model_spec(L=4, att_every=4, H=256)
    ex = ExecSpec(B=4, T=1, R=2, ctx=1024, tiles=TINY_TILES)
    rep = advisor.advise_from_spec(
        ms, ex, chip="tpu7x", moe_mode="grouped_on_chip", dp=4
    )
    assert rep.recommended_scope == "whole_step"
    rep_def = advisor.advise_from_spec(
        ms,
        ExecSpec(B=4, T=1, R=2, ctx=1024),
        chip="tpu7x",
        moe_mode="grouped_on_chip",
        dp=4,
    )
    # 64 MiB physical minus 2 MiB reserve
    expected_vmem_mib = float(constants.DEFAULT_USABLE_VMEM_MIB)
    assert rep_def.recommended_scope == "whole_step"
    assert rep_def.vmem_budget_mib == expected_vmem_mib

    ms_non_periodic = make_fake_model_spec(L=5, att_every=4, H=256)
    rep_np = advisor.advise_from_spec(
        ms_non_periodic, ex, chip="tpu7x", moe_mode="grouped_on_chip", dp=4
    )
    cross = next(s for s in rep_np.scopes if s.scope == "whole_step")
    assert cross.feasible is False
    assert rep_np.fastest_scope == "single_layer"
    assert rep_np.recommended_scope == "per_op"


def test_spec_bridge_jax_isolation_ast() -> None:
    """Verify spec_bridge.py isolates jax and planner imports inside ast.Try."""
    bridge_path = Path(advisor.__file__).with_name("spec_bridge.py")
    tree = ast.parse(bridge_path.read_text("utf-8"), filename=str(bridge_path))
    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            names = [alias.name for alias in node.names]
            mod = getattr(node, "module", "") or ""
            assert not any("jax" in n for n in names), f"Unguarded jax import: {names}"
            assert "jax" not in mod, f"Unguarded jax module import: {mod}"
            assert not any("planner" in n for n in names), (
                f"Unguarded planner import: {names}"
            )
    try_nodes = [n for n in tree.body if isinstance(n, ast.Try)]
    assert any(
        any(
            (
                isinstance(h.type, ast.Tuple)
                and any(
                    getattr(e, "id", "") in ("ImportError", "RuntimeError")
                    for e in h.type.elts
                )
            )
            or getattr(h.type, "id", "") in ("ImportError", "RuntimeError")
            for h in t.handlers
        )
        for t in try_nodes
    ), "spec_bridge.py must have ast.Try catching (ImportError, RuntimeError)"


def test_advise_from_spec_raises_when_planner_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify advise_from_spec raises MegakernelConfigError when planner is missing."""
    monkeypatch.setattr(spec_bridge, "planner", None)
    ms = make_fake_model_spec(L=4, att_every=4, H=256)
    ex = ExecSpec(B=4, T=1, R=2, ctx=1024, tiles=TINY_TILES)
    with pytest.raises(MegakernelConfigError, match="Megakernel planner requires JAX"):
        advisor.advise_from_spec(ms, ex)


def test_qwen35_decode_preset_whole_step_admits_shipped_verify_rows() -> None:
    """Verify whole_step fits 64 verify rows and reports the ideal VMEM ceiling."""
    dep = advisor.DEPLOYMENT_PRESETS["qwen3.5-397b"]
    rows_per_seq = dep.decode.spec_tokens + 1
    rep = advisor.advise(
        advisor.resolve_model_profile("qwen3.5-397b"),
        _preset_workload(
            dep.decode,
            phase="verify",
            batch=64 // rows_per_seq,  # Minimum verify rows for Qwen 3.5
            spec_tokens=dep.decode.spec_tokens,
            moe_mode="grouped_on_chip",
        ),
        "tpu7x",
    )
    whole = next(s for s in rep.scopes if s.scope == "whole_step")
    rows_cap = rows_per_seq * (whole.t_vmem_max // rows_per_seq)
    min_verify_rows = 64  # Minimum verify rows for Qwen 3.5
    assert rows_cap >= min_verify_rows
    assert abs(rows_cap - 448) <= rows_per_seq  # Ideal VMEM rows for Qwen 3.5


@pytest.mark.parametrize("batch", [1, 2, 4, 8])
def test_kimi_k3_decode_preset_low_batch_prefers_whole_step(batch: int) -> None:
    """Verify Kimi-K3 decode at batch 1..8 on the TP32 mesh is one call."""
    dep = advisor.DEPLOYMENT_PRESETS["kimi-k3"]
    rep = advisor.advise(
        advisor.resolve_model_profile("kimi-k3"),
        _preset_workload(
            dep.decode,
            phase="decode",
            batch=batch,
        ),
        "tpu7x",
    )
    assert rep.fastest_scope == "whole_step"
    assert rep.recommended_scope == "per_op"
    assert (
        "rank_count_outside_core_grid"
        in next(s for s in rep.scopes if s.scope == "whole_step").blockers
    )


@pytest.mark.parametrize("window", [2048, 8192])
def test_kimi_k3_prefill_preset_evaluates_layer_group(window: int) -> None:
    """Verify the shipped 13-call layer grouping is feasible and evaluated."""
    dep = advisor.DEPLOYMENT_PRESETS["kimi-k3"]
    rep = advisor.advise(
        advisor.resolve_model_profile("kimi-k3"),
        _preset_workload(
            dep.prefill,
            phase="prefill",
            batch=1,
            isl=window,
            chunk_size=window,
            moe_mode="grouped_on_chip",
        ),
        "tpu7x",
    )
    # 13 declared groups in Kimi-K3 prefill preset
    expected_prefill_calls = len(
        advisor.DEPLOYMENT_PRESETS["kimi-k3"].prefill.layer_groups
    )
    group = next(s for s in rep.scopes if s.scope == "layer_group")
    assert group.feasible
    assert group.ops_per_step == expected_prefill_calls
    assert rep.active_tokens == window // dep.prefill.pcp


def test_layer_groups_must_cover_every_layer() -> None:
    """Verify a layer grouping that does not sum to num_layers is rejected."""
    with pytest.raises(MegakernelConfigError):
        advisor.advise(
            advisor.resolve_model_profile("kimi-k3"),
            advisor.WorkloadSpec(tp=32, ep=32, layer_groups=(8, 8)),
            "tpu7x",
        )


def test_whole_step_tiles_overflow_and_reads_weights_once() -> None:
    """Verify whole_step tiles tokens past the ceiling instead of re-passing."""
    dep = advisor.DEPLOYMENT_PRESETS["qwen3.5-397b"]
    rep = advisor.advise(
        advisor.resolve_model_profile("qwen3.5-397b"),
        _preset_workload(
            dep.decode,
            phase="verify",
            batch=(448 // (dep.decode.spec_tokens + 1))
            + 4,  # Ideal VMEM rows for Qwen 3.5
            spec_tokens=dep.decode.spec_tokens,
            moe_mode="grouped_on_chip",
        ),
        "tpu7x",
    )
    tiled = next(s for s in rep.scopes if s.scope == "whole_step")
    one_l = next(s for s in rep.scopes if s.scope == "single_layer")
    assert tiled.n_splits > 1
    assert tiled.feasible
    assert tiled.w_step_gb == one_l.w_step_gb
    assert tiled.step_bound_us < 1.5 * one_l.step_bound_us


def test_fused_scopes_price_the_recommended_moe_mode() -> None:
    """Verify a full row tile per EP rank prices on-chip grouping either way."""
    prof = advisor.resolve_model_profile("qwen3.5-397b")
    steps = []
    for mode in ("masked_on_chip", "grouped_on_chip"):
        rep = advisor.advise(
            prof,
            advisor.WorkloadSpec(phase="decode", batch=64, tp=8, ep=8, moe_mode=mode),
            "tpu7x",
        )
        steps.append(next(s for s in rep.scopes if s.scope == "whole_step"))
        assert rep.recommended_moe_mode == "grouped_on_chip"
    assert steps[0].moe_flop_factor == steps[1].moe_flop_factor


def test_intra_host_dp_grouped_moe_consistency() -> None:
    """Verify intra-host DPExchange groups MoE while multi-host DCN stays ungrouped."""
    grouped_phi_ceiling = 5.0
    ungrouped_phi_floor = 10.0
    prof = advisor.resolve_model_profile("qwen3.5-397b")
    rep_grp = advisor.advise(
        prof,
        advisor.WorkloadSpec(
            phase="decode",
            batch=600,
            context=32768,
            tp=2,
            dp=4,
            ep=8,
            moe_mode="grouped_on_chip",
        ),
        "tpu7x",
    )
    ws_grp = next(s for s in rep_grp.scopes if s.scope == "whole_step")
    assert ws_grp.moe_flop_factor < grouped_phi_ceiling
    assert rep_grp.fastest_scope == "whole_step"
    assert rep_grp.recommended_scope == "per_op"
    assert rep_grp.recommended_moe_mode == "grouped_on_chip"
    assert float(rep_grp.metrics["achievable_gain_pct"]["value"]) > 0.0

    rep_msk = advisor.advise(
        prof,
        advisor.WorkloadSpec(
            phase="decode",
            batch=600,
            context=32768,
            tp=2,
            dp=4,
            ep=8,
            moe_mode="masked_on_chip",
        ),
        "tpu7x",
    )
    ws_msk = next(s for s in rep_msk.scopes if s.scope == "whole_step")
    assert ws_msk.moe_flop_factor > ungrouped_phi_floor
    assert rep_msk.recommended_moe_mode != "grouped_on_chip"

    rep_dcn = advisor.advise(
        prof,
        advisor.WorkloadSpec(
            phase="decode",
            batch=600,
            context=32768,
            tp=2,
            dp=8,
            ep=16,
            moe_mode="grouped_on_chip",
        ),
        "tpu7x",
    )
    ws_dcn = next(s for s in rep_dcn.scopes if s.scope == "whole_step")
    assert ws_dcn.moe_flop_factor > ungrouped_phi_floor
    assert rep_dcn.recommended_moe_mode != "grouped_on_chip"


def test_unchunked_pp_prefill_preserves_window() -> None:
    """Verify unchunked B=1 prefill under PP>1 preserves its full stage window."""
    prof = advisor.resolve_model_profile("qwen3.5-397b")
    isl_tokens = 4096
    rep_single = advisor.advise(
        prof,
        advisor.WorkloadSpec(
            phase="prefill",
            batch=1,
            isl=isl_tokens,
            tp=1,
            pp=8,
            ep=1,
            chunk_size=0,
            microbatches=0,
        ),
        "tpu7x",
    )
    assert rep_single.active_tokens == isl_tokens
    assert rep_single.recommended_scope == "per_op"
    assert rep_single.recommended_moe_mode == "grouped_in_hbm"
    assert float(rep_single.metrics["compiler_proof_gain_pct"]["value"]) == 0.0

    rep_microbatched = advisor.advise(
        prof,
        advisor.WorkloadSpec(
            phase="prefill",
            batch=8,
            isl=isl_tokens,
            tp=1,
            pp=8,
            ep=1,
            chunk_size=0,
            microbatches=8,
        ),
        "tpu7x",
    )
    assert rep_microbatched.active_tokens == isl_tokens


def test_core_limits_literals_match_megakernel_core() -> None:
    """Verify CORE_LIMITS matches megakernel/core/exec_spec.py and profile.py."""
    mk_dir = Path(advisor.__file__).resolve().parent.parent
    exec_text = (mk_dir / "core" / "exec_spec.py").read_text("utf-8")
    prof_text = (mk_dir / "core" / "profile.py").read_text("utf-8")
    tiles_default = Tiles()
    lim = advisor.CORE_LIMITS
    assert f"MAX_BATCH_ROWS = {lim.max_batch_rows}" in exec_text
    assert "_NUMERICS_GRID_R = (1, 2, 4, 8, 16)" in prof_text

    assert lim.max_batch_rows == constants.CORE_LIMITS.max_batch_rows
    assert lim.verified_ranks == constants.CORE_LIMITS.verified_ranks
    assert lim.chip_kinds == constants.CORE_LIMITS.chip_kinds
    assert lim.max_ar_slots_per_layer == constants.CORE_LIMITS.max_ar_slots_per_layer
    assert (lim.nbd, lim.nbe, lim.bank, lim.ebank, lim.vmem_mib) == (
        tiles_default.nbd,
        tiles_default.nbe,
        tiles_default.bank,
        tiles_default.ebank,
        tiles_default.vmem_mib,
    )


def test_unknown_model_type_is_hypothetical() -> None:
    """Verify an unrecognized HF model_type is hypothetical and never recommended."""
    cfg = {
        "model_type": "brand_new_2027_arch",
        "num_hidden_layers": 32,
        "hidden_size": 4096,
        "intermediate_size": 11008,
        "num_attention_heads": 32,
        "num_key_value_heads": 8,
        "head_dim": 128,
    }
    with pytest.raises(advisor.UnsupportedArchitectureError):
        advisor.model_profile_from_hf_config(cfg, name="brand-new-2027")

    prof = advisor.ModelProfile(
        name="brand-new-2027",
        model_type="brand_new_2027_arch",
        num_layers=32,
        num_attention_layers=32,
        num_moe_layers=0,
        hidden_size=4096,
        intermediate_size=11008,
        num_attention_heads=32,
        num_kv_heads=8,
        head_dim=128,
        weight_dtype="fp8",
        expert_dtype="fp8",
        architecture_unverified=True,
    )
    rep = advisor.advise(
        prof, advisor.WorkloadSpec(phase="decode", batch=16, tp=8), "tpu7x"
    )
    whole = next(s for s in rep.scopes if s.scope == "whole_step")
    assert rep.recommended_scope == "per_op"
    assert rep.fastest_scope == "whole_step"
    assert whole.support == "hypothetical"
    assert "model_type_unrecognized" in whole.blockers
    assert rep.hypothetical_best is not None
    assert rep.hypothetical_best["scope"] == "whole_step"


@pytest.mark.parametrize(
    ("workload", "chip", "expected_blocker"),
    [
        (
            advisor.WorkloadSpec(phase="decode", batch=256, tp=8, ep=8, dp=1),
            "tpu7x",
            "rows_exceed_core_max_batch_rows",
        ),
        (
            advisor.WorkloadSpec(phase="decode", batch=64, tp=2, ep=8, dp=4),
            "tpu7x",
            "all_to_all_not_in_core",
        ),
        (
            advisor.WorkloadSpec(phase="decode", batch=16, tp=32, ep=32, dp=1),
            "tpu7x",
            "rank_count_outside_core_grid",
        ),
        (
            advisor.WorkloadSpec(phase="decode", batch=16, tp=16, ep=16, dp=1),
            "v6e",
            "chip_without_core_budget",
        ),
    ],
)
def test_no_fused_recommendation_outside_core_limits(
    workload: advisor.WorkloadSpec,
    chip: str,
    expected_blocker: str,
) -> None:
    """Verify >128 rows, ep_dp>1, 32 ranks, or non-TPU7x chips gate to per_op."""
    prof = advisor.resolve_model_profile("qwen3.5-397b")
    rep = advisor.advise(prof, workload, chip)
    whole = next(s for s in rep.scopes if s.scope == "whole_step")
    assert whole.support == "hypothetical"
    assert expected_blocker in whole.blockers
    assert rep.recommended_scope == "per_op"
    assert rep.build_megakernel is False
    assert rep.fastest_scope in ("whole_step", "single_layer", "layer_group")
    assert rep.hypothetical_best is not None


def test_in_slice_all_reduce_uses_ici_for_every_rank_count() -> None:
    """Verify TP32 All-Reduce uses single-slice ICI rather than DCN (Ledger Row 2)."""
    chip = advisor.resolve_chip_spec("tpu7x")
    msg_bytes = 64 * 8192 * 2
    t_trans, t_lat, hops, _ = advisor.compute_all_reduce_transfer_cost(
        msg_bytes, 32, chip
    )
    expected_hops = 1 + 2 + 2  # 5 hops across 2x2x4 3D torus for 32 cores (16 chips)
    assert hops == expected_hops
    # 2 * (31 / 32) * 1 MiB over one 100 GB/s link is 20.3162 us, plus 5 hops of 1 us.
    expected_ar_cost_us = round(
        (
            2.0 * (31.0 / 32.0) * msg_bytes / chip.ici_link_bw_bytes_per_s
            + hops * chip.collective_latency_s
        )
        * 1e6,
        4,
    )
    assert round((t_trans + t_lat) * 1e6, 4) == expected_ar_cost_us


def test_all_reduce_transfer_uses_single_link() -> None:
    """Verify all-reduce transfer is priced on one ICI link, not the egress sum."""
    chip = advisor.resolve_chip_spec("tpu7x")
    msg_bytes = 64 * 8192 * 2
    t_trans, _, _, volume = advisor.compute_all_reduce_transfer_cost(msg_bytes, 8, chip)
    assert t_trans == pytest.approx(volume / chip.ici_link_bw_bytes_per_s)
    assert t_trans > volume / chip.ici_egress_bw_bytes_per_s


@pytest.mark.parametrize("batch", [16, 512])
def test_strict_floor_prices_all_reduce_on_egress(batch: int) -> None:
    """Verify the strict floor ignores link bandwidth while the floor pays it."""
    prof = advisor.resolve_model_profile("qwen3.5-397b")
    wl = advisor.WorkloadSpec(phase="decode", batch=batch, tp=8, ep=8)
    chip = advisor.resolve_chip_spec("tpu7x")
    slow_chip = dataclasses.replace(
        chip, ici_link_bw_bytes_per_s=chip.ici_link_bw_bytes_per_s / 2
    )
    fast, slow = (
        next(
            s for s in advisor.advise(prof, wl, spec).scopes if s.scope == "whole_step"
        )
        for spec in (chip, slow_chip)
    )
    multi_pass_batch = 512
    assert (fast.n_splits > 1) == (batch == multi_pass_batch)
    assert slow.step_floor_strict_us == pytest.approx(fast.step_floor_strict_us)
    assert slow.step_bound_us > fast.step_bound_us


def test_all_to_all_fanout_capped_by_destination_groups() -> None:
    """Verify MoE All-to-All fanout caps at min(top_k, ep_dp) (Ledger Row 3)."""
    prof = advisor.resolve_model_profile("qwen3.5-397b")
    chip = advisor.resolve_chip_spec("tpu7x")
    wl = advisor.WorkloadSpec(
        phase="decode", batch=2400, tp=2, dp=4, ep=8, moe_mode="grouped_on_chip"
    )
    ctx = _build_scope_context(prof, wl, chip)
    assert ctx.ep_dp == wl.ep // wl.tp
    # A2A transfer priced at egress bandwidth across destination groups.
    expected_a2a_us = (
        2.0
        * (ctx.ep_dp - 1)
        / ctx.ep_dp
        * min(max(1, prof.top_k), ctx.ep_dp)
        * ctx.token_residual_bytes
        / chip.ici_egress_bw_bytes_per_s
    ) * 1e6
    assert ctx.a2a_time * 1e6 == pytest.approx(expected_a2a_us, abs=0.01)


def test_strict_floor_not_above_uniform_estimate() -> None:
    """Verify strict balanced routing floor vs uniform coupon-collector (Row 4)."""
    prof = advisor.resolve_model_profile("qwen3.5-397b")
    wl = advisor.WorkloadSpec(phase="decode", batch=16, tp=8, ep=8)
    rep = advisor.advise(
        prof,
        wl,
        "tpu7x",
    )
    whole = next(s for s in rep.scopes if s.scope == "whole_step")
    chip = advisor.resolve_chip_spec("tpu7x")
    # Strict floor weights for Qwen3.5-397B are non-zero and below total weights.
    assert 0.0 < whole.w_step_strict_gb < whole.w_step_gb
    # The strict floor incorporates routed compute chain, egress collectives,
    # and pipeline fill/drain. It strictly exceeds the pure HBM stream time
    # of strict weights (586.64 us) and sits below step_bound_us (4192.01 us).
    raw_hbm_floor_us = round(
        ((whole.w_step_strict_gb * 1e9) / chip.hbm_bw_bytes_per_s) * 1e6, 2
    )
    assert whole.step_floor_strict_us > raw_hbm_floor_us
    assert 0.0 < whole.step_floor_strict_us < whole.step_bound_us


def test_fused_core_is_max_of_hbm_and_compute_plus_comm() -> None:
    """Verify fused critical path uses max(hbm_time_s, t_matmul + comm_s)."""
    prof = advisor.resolve_model_profile("qwen3.5-0.8b")
    rep = advisor.advise(
        prof,
        advisor.WorkloadSpec(phase="prefill", batch=1, isl=512, chunk_size=512, tp=8),
        "tpu7x",
    )
    whole = next(s for s in rep.scopes if s.scope == "whole_step")
    assert whole.compute_us + whole.comm_us > whole.hbm_us
    assert whole.comm_us > 0.0
    assert whole.roof_floor_us == pytest.approx(
        max(whole.hbm_us, whole.compute_us + whole.comm_us) + whole.fill_drain_us,
        abs=0.05,
    )
    assert (
        whole.roof_floor_us
        > max(whole.compute_us, whole.hbm_us, whole.comm_us) + whole.fill_drain_us
    )


def test_core_time_routed_stream_cannot_hide_collectives() -> None:
    """Verify collectives add to the routed-expert chain, not under the HBM stream."""
    t_dense_s = 0.05
    t_moe_s = 0.10
    routed_hbm_s = 0.80
    hbm_s = 1.00
    comm_s = 0.30
    chain_s = _routed_chain(t_dense_s, routed_hbm_s, t_moe_s)
    assert chain_s == pytest.approx(t_dense_s + routed_hbm_s)
    fused_s = _core_time(hbm_s, chain_s, comm_s, fused=True)
    assert fused_s == pytest.approx(chain_s + comm_s)
    assert fused_s > max(hbm_s, t_dense_s + t_moe_s + comm_s)
    split_s = _core_time(hbm_s, chain_s, comm_s, fused=False)
    assert split_s == pytest.approx(hbm_s + comm_s)
    assert _core_time(hbm_s, chain_s, 0.0, fused=True) == hbm_s


def test_whole_step_bound_pays_collectives_after_router() -> None:
    """Verify a slower collective raises the fused MoE floor one for one."""
    prof = advisor.resolve_model_profile("qwen3.5-397b")
    wl = advisor.WorkloadSpec(
        phase="verify", batch=1, spec_tokens=3, context=4096, tp=8, dcp=4, ep=8
    )
    chip = advisor.resolve_chip_spec("tpu7x")
    slow_collective_us = 3.0
    slow_chip = dataclasses.replace(
        chip,
        collective_latency_s=slow_collective_us * 1e-6,
        collective_latency_us_bracket=(slow_collective_us, slow_collective_us),
    )
    wholes = [
        next(
            s for s in advisor.advise(prof, wl, spec).scopes if s.scope == "whole_step"
        )
        for spec in (chip, slow_chip)
    ]
    fast, slow = wholes
    fast_core_us = fast.roof_floor_us - fast.fill_drain_us
    slow_core_us = slow.roof_floor_us - slow.fill_drain_us
    assert fast.n_splits == slow.n_splits == 1
    assert fast_core_us > fast.hbm_us
    assert slow.comm_us > fast.comm_us
    assert slow_core_us - fast_core_us == pytest.approx(
        slow.comm_us - fast.comm_us, abs=0.05
    )


def test_achievable_upper_band_never_inverts_below_achievable_or_floor() -> None:
    """Verify step_achievable_upper_us (Row 7) and CSA/HC bounds (Rows 8, 9)."""
    prof = advisor.resolve_model_profile("qwen3.5-397b")
    rep = advisor.advise(
        prof,
        advisor.WorkloadSpec(
            phase="verify", batch=16, spec_tokens=3, context=4096, tp=8, dcp=4, ep=8
        ),
        "tpu7x",
    )
    whole = next(s for s in rep.scopes if s.scope == "whole_step")
    assert whole.step_achievable_upper_us >= whole.step_achievable_us
    for sc in rep.scopes:
        assert sc.step_achievable_upper_us >= max(
            sc.step_achievable_us, sc.step_upper_us
        )

    p_ds4 = advisor.resolve_model_profile("deepseek-v4")
    c7x = advisor.resolve_chip_spec("tpu7x")
    wl_ds4 = advisor.WorkloadSpec(
        phase="decode", batch=128, context=160000, dp=16, tp=1, ep=16
    )
    ctx_ds4 = _build_scope_context(p_ds4, wl_ds4, c7x)
    expected_ds4_kv_read_gb = (
        p_ds4.num_layers
        * (wl_ds4.batch // wl_ds4.dp)
        * wl_ds4.context
        * (2 * p_ds4.num_kv_heads * p_ds4.head_dim)
        * constants.DTYPE_BYTES[p_ds4.kv_dtype]
        * p_ds4.kv_compress_factor
        / 1e9
    )
    assert ctx_ds4.kv_cache_read_bytes / 1e9 == pytest.approx(
        expected_ds4_kv_read_gb, abs=0.01
    )

    p_q4 = advisor.resolve_model_profile("qwen4-exp")
    parts_q4 = advisor.compute_per_token_vmem_bytes(
        p_q4, advisor.WorkloadSpec(phase="decode", batch=16, tp=8)
    )
    expected_qwen4_residual_bytes = (
        4 * max(1, p_q4.residual_streams) + 2 + 6
    ) * p_q4.hidden_size
    assert parts_q4["residual_buffer_bytes"] == expected_qwen4_residual_bytes


def test_gain_split_sums_to_total_and_fact_part_ignores_priors() -> None:
    """Verify gain_fact_us and gain_assumed_us split and ASSUMED tag (Row 5)."""
    prof = advisor.resolve_model_profile("qwen3.5-397b")
    rep = advisor.advise(
        prof,
        advisor.WorkloadSpec(
            phase="decode",
            batch=16,
            tp=8,
            dcp=4,
            ep=8,
        ),
        "tpu7x",
    )
    gain_fact_us = float(rep.metrics["gain_fact_us"]["value"])
    gain_assumed_us = float(rep.metrics["gain_assumed_us"]["value"])
    total_gain = float(rep.metrics["compiler_proof_gain_us"]["value"])
    assert gain_fact_us > 0.0
    assert gain_assumed_us > 0.0
    assert total_gain == pytest.approx(gain_fact_us + gain_assumed_us, abs=0.01)
    assert rep.metrics["compiler_proof_gain_us"]["tag"] == "ASSUMED"


def test_invalid_sharding_returns_invalid_config() -> None:
    """Verify non-power-of-two TP produces invalid_config verdict and per_op."""
    prof = advisor.resolve_model_profile("qwen3.5-397b")
    rep = advisor.advise(
        prof,
        advisor.WorkloadSpec(phase="decode", batch=16, tp=3, ep=1, objective="both"),
        "tpu7x",
    )
    assert rep.verdict.startswith("invalid_config(")
    assert rep.fastest_verdict == rep.verdict
    assert rep.recommended_scope == "per_op"
    assert rep.build_megakernel is False


def test_bundled_priors_keep_prior_only_confidence() -> None:
    """Verify architectural_prior bundled measurements keep prior_only confidence."""
    prof = advisor.resolve_model_profile("qwen3.5-397b")
    meas: str | dict[str, object] = (
        "bundled"
        if advisor.chips.public_calibration_path("tpu7x") is not None
        else {
            "schema": "advisor.measurements.v1",
            "chip": "tpu7x",
            "source": {"kind": "architectural_prior"},
        }
    )
    rep = advisor.advise(
        prof,
        advisor.WorkloadSpec(phase="decode", batch=16, tp=8, ep=8),
        "tpu7x",
        measurements=meas,
    )
    assert rep.confidence == "prior_only"


def _verify_whole_step(
    batch: int, max_rows_per_call: int, op_overhead_us: float = 0.0
) -> tuple[advisor.AdvisorReport, advisor.ScopeEstimate]:
    """Advise a Qwen3.5-397B verify step; return the report and its whole_step."""
    rep = advisor.advise(
        advisor.resolve_model_profile("qwen3.5-397b"),
        advisor.WorkloadSpec(
            phase="verify",
            batch=batch,
            spec_tokens=3,  # rows_per_request = 1 prompt + 3 spec = 4
            context=4096,
            tp=8,
            dcp=4,
            ep=8,
            max_rows_per_call=max_rows_per_call,
            op_overhead_us=op_overhead_us,
        ),
        "tpu7x",
    )
    return rep, next(s for s in rep.scopes if s.scope == "whole_step")


def test_max_rows_per_call_clamps_single_call_rows() -> None:
    """Verify a declared row limit caps rows per call without changing the price."""
    _, ideal = _verify_whole_step(16, 0)
    _, capped = _verify_whole_step(16, constants.DEFAULT_ROW_LIMIT)
    assert ideal.t_vmem_max > constants.DEFAULT_ROW_LIMIT
    assert capped.t_vmem_max == constants.DEFAULT_ROW_LIMIT
    assert capped.n_splits == 1
    assert capped.step_bound_us == ideal.step_bound_us


def test_rows_above_single_call_limit_priced_as_microbatches() -> None:
    """Verify rows past the limit cost a full call plus the remainder call."""
    rows_per_req = 4
    full_requests = constants.DEFAULT_ROW_LIMIT // rows_per_req
    split_batch = 28
    _, split = _verify_whole_step(split_batch, constants.DEFAULT_ROW_LIMIT)
    _, full = _verify_whole_step(full_requests, constants.DEFAULT_ROW_LIMIT)
    _, rest = _verify_whole_step(
        split_batch - full_requests, constants.DEFAULT_ROW_LIMIT
    )
    _, tiled = _verify_whole_step(28, 0)
    assert split.n_splits == full.n_splits + rest.n_splits
    assert split.ops_per_step == full.ops_per_step + rest.ops_per_step
    assert split.step_bound_us == pytest.approx(
        full.step_bound_us + rest.step_bound_us, abs=0.02
    )
    assert split.step_achievable_us == pytest.approx(
        full.step_achievable_us + rest.step_achievable_us, abs=0.02
    )
    assert split.w_step_gb == pytest.approx(full.w_step_gb + rest.w_step_gb, abs=1e-3)
    assert split.step_bound_us > tiled.step_bound_us


def test_max_rows_per_call_splits_whole_requests() -> None:
    """Verify a limit between request boundaries rounds down to whole requests."""
    _, exact = _verify_whole_step(28, constants.DEFAULT_ROW_LIMIT)
    _, ragged = _verify_whole_step(28, constants.DEFAULT_ROW_LIMIT + 1)
    assert ragged.t_vmem_max == constants.DEFAULT_ROW_LIMIT
    assert ragged == exact


def test_max_rows_per_call_below_one_request_is_config_error() -> None:
    """Verify a limit smaller than one verify request is rejected."""
    with pytest.raises(MegakernelConfigError, match="below one verify request"):
        _verify_whole_step(16, 4 - 1)


def test_microbatched_whole_step_keeps_support_status() -> None:
    """Verify a split whole_step stays supported and is flagged, not blocked."""
    rep, split = _verify_whole_step(28, constants.DEFAULT_ROW_LIMIT)
    assert split.support == "within_core_limits"
    assert split.blockers == ()
    assert "microbatch_rerun" in rep.flags
    assert "multi_call_unmodeled" not in rep.flags
    rep_one, _ = _verify_whole_step(16, constants.DEFAULT_ROW_LIMIT)
    assert "microbatch_rerun" not in rep_one.flags


def test_microbatched_step_intensity_counts_weights_once() -> None:
    """Verify two equal full calls keep the arithmetic intensity of one call."""
    favoring_gap_us = 500.0
    rows_per_req = 4
    full_requests = constants.DEFAULT_ROW_LIMIT // rows_per_req
    two_calls = 2
    one, _ = _verify_whole_step(
        full_requests, constants.DEFAULT_ROW_LIMIT, favoring_gap_us
    )
    two, split = _verify_whole_step(
        two_calls * full_requests,
        constants.DEFAULT_ROW_LIMIT,
        favoring_gap_us,
    )
    assert one.fastest_scope == two.fastest_scope == "whole_step"
    assert split.n_splits == two_calls
    # Doubling the batch doubles both the FLOPs and the bytes of the step.
    assert two.step_arithmetic_intensity == pytest.approx(
        one.step_arithmetic_intensity, abs=0.02
    )


def test_in_call_passes_stay_flagged_under_row_limit() -> None:
    """Verify a call larger than the VMEM ceiling keeps the unmodeled-pass flag."""
    favoring_gap_us = 500.0
    rows_per_req = 4
    two_calls = 2
    qwen35_ideal_vmem_rows = 448
    limit = two_calls * qwen35_ideal_vmem_rows
    requests = (limit + qwen35_ideal_vmem_rows) // rows_per_req
    rep, split = _verify_whole_step(requests, limit, favoring_gap_us)
    assert rep.fastest_scope == "whole_step"
    # One full call of two ideal-ceiling passes, then a remainder call of one.
    assert split.n_splits == two_calls + 1
    assert {"microbatch_rerun", "multi_call_unmodeled"} <= set(rep.flags)
    single_pass_calls, _ = _verify_whole_step(
        two_calls * (constants.DEFAULT_ROW_LIMIT // rows_per_req),
        constants.DEFAULT_ROW_LIMIT,
        favoring_gap_us,
    )
    assert single_pass_calls.fastest_scope == "whole_step"
    assert "multi_call_unmodeled" not in single_pass_calls.flags


def test_microbatch_zero_map_is_sum_of_calls() -> None:
    """Verify overridden latency priors reach every call of a split whole_step."""
    prof = advisor.resolve_model_profile("qwen3.5-397b")
    chip = advisor.resolve_chip_spec("tpu7x")

    def zero_latency_bound(batch: int) -> float:
        """Return the whole_step bound with every latency prior set to zero."""
        wl = advisor.WorkloadSpec(
            phase="verify",
            batch=batch,
            spec_tokens=3,  # rows_per_request = 1 prompt + 3 spec = 4
            context=4096,
            tp=8,
            dcp=4,
            ep=8,
            max_rows_per_call=constants.DEFAULT_ROW_LIMIT,
        )
        ctx = dataclasses.replace(
            _build_scope_context(prof, wl, chip),
            lat_coll=0.0,
            lat_coll_upper=0.0,
            ar_base_lat_s=0.0,
            ar_inter_latency_s=0.0,
            launch_gap_upper_us=0.0,
        )
        estimates, _ = _evaluate_scopes(ctx, prof, wl)
        return next(e for e in estimates if e.scope == "whole_step").step_bound_us

    rows_per_req = 4
    full_requests = constants.DEFAULT_ROW_LIMIT // rows_per_req
    split_batch = 28
    split_us = zero_latency_bound(split_batch)
    assert split_us == pytest.approx(
        zero_latency_bound(full_requests)
        + zero_latency_bound(split_batch - full_requests),
        abs=0.02,
    )
    _, priced = _verify_whole_step(split_batch, constants.DEFAULT_ROW_LIMIT)
    assert split_us < priced.step_bound_us


def test_declared_preset_status_reachable(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify a deployments.json megakernel declaration yields declared support."""
    dep = advisor.DEPLOYMENT_PRESETS["qwen3.5-397b"]
    declaration = {
        "source": "unit test",
        "scopes": ["whole_step"],
        "max_rows": constants.DEFAULT_ROW_LIMIT,
        "ranks": dep.decode.tp,
    }
    declared = dataclasses.replace(
        dep, decode=dataclasses.replace(dep.decode, megakernel=declaration)
    )
    monkeypatch.setattr(support, "DEPLOYMENT_PRESETS", {"qwen3.5-397b": declared})
    _, one_call = _verify_whole_step(16, 0)
    assert one_call.support == "declared"
    assert one_call.t_vmem_max == constants.DEFAULT_ROW_LIMIT
    rep, split = _verify_whole_step(28, 0)
    assert split.support == "declared"
    assert split.n_splits > 1
    assert "microbatch_rerun" in rep.flags


def test_preset_row_limit_below_one_request_is_config_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify a preset row limit below one verify request is rejected, not ignored."""
    dep = advisor.DEPLOYMENT_PRESETS["qwen3.5-397b"]
    declaration = {
        "source": "unit test",
        "scopes": ["whole_step"],
        "max_rows": 4 - 1,
        "ranks": dep.decode.tp,
    }
    declared = dataclasses.replace(
        dep, decode=dataclasses.replace(dep.decode, megakernel=declaration)
    )
    monkeypatch.setattr(support, "DEPLOYMENT_PRESETS", {"qwen3.5-397b": declared})
    with pytest.raises(MegakernelConfigError, match="below one verify request"):
        _verify_whole_step(16, 0)


def test_validate_workload_names_row_limit_below_one_request() -> None:
    """Verify the non-raising validator reports a row limit below one request."""
    rows_per_req = 4
    workload = advisor.WorkloadSpec(
        phase="verify",
        batch=16,
        spec_tokens=rows_per_req - 1,
        context=4096,
        tp=8,
        dcp=4,
        ep=8,
        max_rows_per_call=rows_per_req - 1,
    )
    reasons = advisor.validate_workload("qwen3.5-397b", "tpu7x", workload)
    assert reasons == ("row_limit_below_request",)
    one_request = dataclasses.replace(workload, max_rows_per_call=rows_per_req)
    assert advisor.validate_workload("qwen3.5-397b", "tpu7x", one_request) == ()


def test_rows_above_default_buckets_flagged() -> None:
    """Verify CORE_LIMITS mirrors the core bucket defaults and flags larger calls."""
    core_dir = Path(advisor.__file__).resolve().parent.parent / "core"
    policy_text = (core_dir / "runtime" / "policy.py").read_text("utf-8")
    tops = [
        max(
            ast.literal_eval(
                policy_text.split(f"{name}: tuple[int, ...] = ", 1)[1].split("\n", 1)[0]
            )
        )
        for name in ("DEFAULT_BUCKETS", "DEFAULT_VERIFY_BUCKETS")
    ]
    lim = advisor.CORE_LIMITS
    assert [lim.default_decode_bucket_top, lim.default_verify_bucket_top] == tops

    prof = advisor.resolve_model_profile("qwen3.5-397b")
    for batch, flagged in (
        (lim.default_decode_bucket_top, False),
        (lim.default_decode_bucket_top + 1, True),
    ):
        rep = advisor.advise(
            prof, advisor.WorkloadSpec(phase="decode", batch=batch, tp=8, ep=8), "tpu7x"
        )
        assert ("above_default_buckets" in rep.flags) is flagged
    verify_top_requests = lim.default_verify_bucket_top // 4
    rep_at_top, _ = _verify_whole_step(verify_top_requests, 0)
    rep_above, _ = _verify_whole_step(verify_top_requests + 1, 0)
    assert "above_default_buckets" not in rep_at_top.flags
    assert "above_default_buckets" in rep_above.flags


def test_ffn_scratch_includes_fp32_bytes() -> None:
    """Verify FFN scratch uses FP32 bytes so Qwen3.5-397B VMEM cap is 448 rows."""
    prof = advisor.resolve_model_profile("qwen3.5-397b")
    wl = advisor.WorkloadSpec(phase="decode", batch=16, tp=8, ep=8)
    chip = advisor.resolve_chip_spec("tpu7x")
    ctx = _build_scope_context(prof, wl, chip)
    assert ctx.ffn_scratch == (
        constants.DOUBLE_BUFFER_SLOTS
        * constants.ATTN_TILE_TOKENS
        * prof.moe_intermediate_size
        * constants.FP32_BYTES
    )
    rep = advisor.advise(prof, wl, chip)
    whole = next(s for s in rep.scopes if s.scope == "whole_step")
    qwen35_ideal_vmem_rows = 448
    assert whole.t_vmem_max == qwen35_ideal_vmem_rows


def test_physical_config_rejects_undersized_slice_and_invalid_combos() -> None:
    """Verify undersized slices are flagged and unphysical combos raise cleanly."""
    undersized = advisor.validate_workload(
        "qwen3.5-397b",
        "tpu7x",
        advisor.WorkloadSpec(phase="decode", batch=4, tp=2, ep=2, dp=1),
    )
    assert "slice_topology_unsupported" in undersized
    valid_slice = advisor.validate_workload(
        "qwen3.5-397b",
        "tpu7x",
        advisor.WorkloadSpec(phase="decode", batch=4, tp=2, ep=2, dp=4),
    )
    assert "slice_topology_unsupported" not in valid_slice

    prof = advisor.resolve_model_profile("qwen3.5-397b")
    tpu7x_vmem_mib = 62.0
    invalid_workloads = (
        advisor.WorkloadSpec(phase="decode", batch=2, tp=2, dp=4, ep=8),
        advisor.WorkloadSpec(phase="verify", batch=4, tp=8, ep=8, spec_tokens=0),
        advisor.WorkloadSpec(
            phase="decode", batch=4, tp=8, ep=8, hbm_utilization_cap=1.5
        ),
        advisor.WorkloadSpec(phase="decode", batch=4, tp=8, ep=8, dma_buffers=4),
        advisor.WorkloadSpec(
            phase="decode",
            batch=4,
            tp=8,
            ep=8,
            vmem_budget_mib=tpu7x_vmem_mib + 1.0,
        ),
    )
    for bad_wl in invalid_workloads:
        with pytest.raises(MegakernelConfigError):
            advisor.advise(prof, bad_wl, "tpu7x")

    kvless = dataclasses.replace(prof, num_kv_heads=0)
    kv_reasons = advisor.validate_workload(
        kvless,
        "tpu7x",
        advisor.WorkloadSpec(phase="decode", batch=8, tp=8, kv_split=2),
    )
    assert "kv_heads_not_divisible_by_tp" in kv_reasons


def test_advisor_core_limits_match_core_package() -> None:
    """Verify spec_bridge AST drift check reports zero differences vs core/."""
    assert spec_bridge.verify_core_limits_sync() == []


def test_per_op_vmem_overflow_marks_infeasible() -> None:
    """Verify static VMEM overflow on per_op marks the baseline infeasible."""
    prof = advisor.resolve_model_profile("qwen3.5-397b")
    rep = advisor.advise(
        prof,
        advisor.WorkloadSpec(
            phase="decode",
            tp=8,
            ep=8,
            batch=4,
            vmem_budget_mib=float(constants.MIN_VMEM_MIB),
        ),
        "tpu7x",
    )
    per_op = next(s for s in rep.scopes if s.scope == "per_op")
    assert (per_op.feasible, per_op.t_vmem_max, per_op.reason) == (
        False,
        0,
        "per_op_vmem_overflow",
    )


def test_recommended_scope_metrics_and_hbm_oom_zero_tps() -> None:
    """Verify summary metrics follow recommended scope and zero out on HBM OOM."""
    prof_bad = advisor.resolve_model_profile("qwen3.5-0.8b")
    rep_bad = advisor.advise(
        prof_bad, advisor.WorkloadSpec(phase="decode", batch=16, tp=2), "tpu7x"
    )
    assert rep_bad.verdict == "invalid_config(slice_topology_unsupported)"
    per_op_bad = next(s for s in rep_bad.scopes if s.scope == "per_op")
    assert float(rep_bad.metrics["step_floor_us"]["value"]) == pytest.approx(
        per_op_bad.step_bound_us, abs=0.01
    )
    assert float(rep_bad.metrics["throughput_tok_s_per_chip"]["value"]) == 0.0
    cmp_res = advisor.verdict.compare_parallelism_candidates(
        "qwen3.5-0.8b",
        (advisor.ParallelLayout(tp=2, ep=1), advisor.ParallelLayout(tp=8, ep=1)),
        chip="tpu7x",
    )
    assert cmp_res["best_step_layout"] == "TP8"
    assert cmp_res["best_throughput_layout"] == "TP8"
    assert cmp_res["verdict"] == "TP8"

    pk3 = advisor.resolve_model_profile("kimi-k3")
    rep_k3 = advisor.advise(
        pk3,
        advisor.WorkloadSpec(phase="decode", batch=4, tp=32, ep=32),
        "tpu7x",
    )
    assert rep_k3.fastest_scope == "whole_step"
    assert rep_k3.recommended_scope == "per_op"
    per_op_k3 = next(s for s in rep_k3.scopes if s.scope == "per_op")
    assert float(rep_k3.metrics["step_floor_us"]["value"]) == pytest.approx(
        per_op_k3.step_bound_us, abs=0.01
    )
    assert float(rep_k3.metrics["throughput_tok_s_per_chip"]["value"]) > 0.0

    p397 = advisor.resolve_model_profile("qwen3.5-397b")
    rep_oom = advisor.advise(
        p397,
        advisor.WorkloadSpec(phase="decode", batch=4, tp=2, ep=2, dp=2),
        "tpu7x",
    )
    assert rep_oom.hbm_fits is False
    assert float(rep_oom.metrics["throughput_tok_s_per_chip"]["value"]) == 0.0


@pytest.mark.parametrize(
    ("model", "workload"),
    [
        (
            "qwen3.5-397b",
            advisor.WorkloadSpec(
                phase="verify", batch=4, spec_tokens=3, context=4096, tp=8, dcp=4, ep=8
            ),
        ),
        ("qwen3.5-397b", advisor.WorkloadSpec(phase="decode", batch=64, tp=8, ep=8)),
        (
            "qwen3.5-0.8b",
            advisor.WorkloadSpec(
                phase="prefill", batch=1, isl=512, chunk_size=512, tp=8
            ),
        ),
    ],
)
def test_scope_subterms_reconstruct_roof_floor(
    model: str, workload: advisor.WorkloadSpec
) -> None:
    """Verify chain_us with hbm_us and comm_us rebuilds every scope's floor."""
    rep = advisor.advise(advisor.resolve_model_profile(model), workload, "tpu7x")
    for sc in rep.scopes:
        core_us = (
            max(sc.hbm_us, sc.chain_us + sc.comm_us)
            if sc.scope in ("whole_step", "layer_group")
            else max(sc.hbm_us, sc.chain_us) + sc.comm_us
        )
        assert sc.chain_us >= sc.compute_us - 0.02
        assert core_us + sc.fill_drain_us == pytest.approx(sc.roof_floor_us, abs=0.02)


def test_vector_unit_fp4_dequant_floor() -> None:
    """Verify vector unit floor on mxfp4 MoE with high FLOP and bandwidth rates."""
    base_prof = advisor.resolve_model_profile("qwen3.5-397b")
    prof = dataclasses.replace(base_prof, expert_dtype="mxfp4")
    workload = advisor.WorkloadSpec(phase="decode", batch=16, tp=8, ep=8)
    base_chip = advisor.resolve_chip_spec("tpu7x")
    high_flops = 1e20
    high_bw = 1e20
    target_vec_rate = 1e12
    chip = dataclasses.replace(
        base_chip,
        fp8_flops_per_s=high_flops,
        bf16_flops_per_s=high_flops,
        hbm_bw_bytes_per_s=high_bw,
        vector_f32_ops_per_s=target_vec_rate,
    )
    rep = advisor.advise(prof, workload, chip)
    sc = next(s for s in rep.scopes if s.scope == "whole_step")
    min_vector_floor_us = 20000.0
    assert sc.step_bound_us >= min_vector_floor_us

    # B1: verify moe_tokens_local scaling in _compute_scope_matmuls
    ctx = _build_scope_context(prof, workload, chip)
    res_1x = _compute_scope_matmuls(
        ctx, prof, workload, 1.0, 10.0 * 1e9, bf16_attention_upper=False
    )
    # B2: verify per_op / single_layer scaling of streamed weight params with n_splits
    res_2x = _compute_scope_matmuls(
        ctx, prof, workload, 1.0, 20.0 * 1e9, bf16_attention_upper=False
    )
    # Doubling weight bytes from 10 GB to 20 GB doubles streamed weight time
    expected_ratio = (20.0 * 1e9) / (10.0 * 1e9)
    assert res_2x[3] / res_1x[3] == pytest.approx(expected_ratio, rel=1e-2)

    # B3: verify strict floor incorporates vector unit terms and is bounded
    min_strict_floor_us = 1000.0
    assert sc.step_floor_strict_us >= min_strict_floor_us
    assert sc.step_floor_strict_us <= sc.step_bound_us

    # D1: verify step_achievable_us derates vector ops by eta_dense
    wl_derated = advisor.WorkloadSpec(
        phase="decode",
        batch=16,
        tp=8,
        ep=8,
        efficiency=advisor.specs.EfficiencySpec(
            eta_hbm=1.0, eta_dense=0.5, eta_moe_grouped=1.0, eta_ici=1.0
        ),
    )
    rep_derated = advisor.advise(prof, wl_derated, chip)
    sc_derated = next(s for s in rep_derated.scopes if s.scope == "whole_step")
    assert sc_derated.step_achievable_us > sc.step_bound_us
    assert sc.compute_us >= min_vector_floor_us


def test_gdn_linear_head_divisibility_reasons() -> None:
    """Verify linear attention head divisibility rejection and replication rules."""
    base_prof = advisor.resolve_model_profile("qwen3.5-0.8b")
    bad_v_prof = dataclasses.replace(
        base_prof, linear_value_heads=18, linear_key_heads=16
    )
    wl_tp16 = advisor.WorkloadSpec(phase="decode", batch=1, tp=16)
    c7x = advisor.resolve_chip_spec("tpu7x")
    reasons_v = _mesh_divisibility_reasons(bad_v_prof, wl_tp16, c7x)
    expected_v_reason = "linear_v_heads_not_divisible_by_tp"
    assert expected_v_reason in reasons_v

    bad_k_prof = dataclasses.replace(
        base_prof, linear_value_heads=16, linear_key_heads=12
    )
    reasons_k = _mesh_divisibility_reasons(bad_k_prof, wl_tp16, c7x)
    expected_k_reason = "linear_k_heads_not_divisible_by_tp"
    assert expected_k_reason in reasons_k

    rep_prof = dataclasses.replace(
        base_prof, linear_value_heads=128, linear_key_heads=16
    )
    wl_tp32 = advisor.WorkloadSpec(phase="decode", batch=1, tp=32)
    reasons_rep = _mesh_divisibility_reasons(rep_prof, wl_tp32, c7x)
    assert expected_v_reason not in reasons_rep
    assert expected_k_reason not in reasons_rep


def test_pp_with_pcp_disagg_prefill_supported() -> None:
    """Verify disaggregated prefill with simultaneous PP and PCP evaluates cleanly."""
    prof = advisor.resolve_model_profile("qwen3.5-397b")
    wl = advisor.WorkloadSpec(
        phase="prefill",
        batch=4,
        isl=4096,
        tp=1,
        pp=4,
        pcp=8,
        ep=8,
    )
    c7x = advisor.resolve_chip_spec("tpu7x")
    rep = advisor.advise(prof, wl, c7x)
    unsupported_reason = "pp_with_pcp_unsupported"
    assert rep.hbm_fits is True
    for sc in rep.scopes:
        assert unsupported_reason not in sc.blockers


def test_tiled_whole_step_arithmetic_intensity_streams_weights_once() -> None:
    """Verify tiled whole_step arithmetic intensity streams weights once."""
    prof = advisor.resolve_model_profile("qwen3.5-397b")
    wl = advisor.WorkloadSpec(
        phase="decode",
        batch=128,
        tp=8,
        ep=8,
        vmem_budget_mib=20,
    )
    c7x = advisor.resolve_chip_spec("tpu7x")
    rep = advisor.advise(prof, wl, c7x)
    best = next(s for s in rep.scopes if s.scope == "whole_step")
    min_splits = 1
    assert best.n_splits > min_splits
    ctx = _build_scope_context(prof, wl, c7x)
    state_bytes = ctx.kv_cache_read_bytes + ctx.recurrent_state_rw_bytes
    weight_reads = 1.0
    b_wa = best.w_step_gb * 1e9 * weight_reads + best.act_hbm_mb * 1e6
    tot_bytes = max(1.0, b_wa + state_bytes)
    total_flops = ctx.flops_dense + best.moe_flop_factor * ctx.flops_moe
    expected_ai = round(total_flops / tot_bytes, 2)
    assert rep.step_arithmetic_intensity == pytest.approx(expected_ai, abs=0.05)


def test_evaluate_corners_matches_step_upper_us() -> None:
    """Verify _evaluate_corners high corner matches step_upper_us."""
    prof = advisor.resolve_model_profile("qwen3.5-397b")
    wl = advisor.WorkloadSpec(phase="decode", batch=16, tp=8, ep=8)
    c7x = advisor.resolve_chip_spec("tpu7x")
    rep = advisor.advise(prof, wl, c7x)
    ctx = _build_scope_context(prof, wl, c7x)
    priors = _resolve_calibrated_priors(wl, c7x, ctx, None)
    box = _assumed_box(ctx, wl, priors)
    ctx_box, wl_box = _apply_box(ctx, wl, box)
    corners = _evaluate_corners(ctx_box, prof, wl_box, box)
    low_corner = (
        box.launch_gap_us[0],
        round(box.collective_s[0] * 1e6, 3),
        box.attn_dots[0],
    )
    high_corner = (
        box.launch_gap_us[1],
        round(box.collective_s[1] * 1e6, 3),
        box.attn_dots[-1],
    )
    sc = next(s for s in rep.scopes if s.scope == "whole_step")
    corner_tol_us = 0.05
    assert corners[low_corner]["whole_step"] == pytest.approx(
        sc.step_bound_us, abs=corner_tol_us
    )
    assert corners[high_corner]["whole_step"] == pytest.approx(
        sc.step_upper_us, abs=corner_tol_us
    )


def test_achievable_upper_scales_a2a_extra_by_eta_ici() -> None:
    """Verify achievable upper scales a2a_extra_s by 1 / eta_ici when ep_dp > 1."""
    prof = advisor.resolve_model_profile("qwen3.5-397b")
    wl_base = advisor.WorkloadSpec(
        phase="decode",
        batch=16,
        tp=2,
        dp=4,
        ep=8,
        efficiency=advisor.specs.EfficiencySpec(1.0, 1.0, 1.0, 1.0),
    )
    c7x = advisor.resolve_chip_spec("tpu7x")
    rep_base = advisor.advise(prof, wl_base, c7x)
    sc_base = next(s for s in rep_base.scopes if s.scope == "whole_step")
    base_tol_us = 0.05
    assert sc_base.step_achievable_upper_us == pytest.approx(
        sc_base.step_upper_us, abs=base_tol_us
    )

    wl_low_ici = dataclasses.replace(
        wl_base, efficiency=advisor.specs.EfficiencySpec(1.0, 1.0, 1.0, 0.7)
    )
    rep_low_ici = advisor.advise(prof, wl_low_ici, c7x)
    sc_low_ici = next(s for s in rep_low_ici.scopes if s.scope == "whole_step")
    min_increase_us = 200.0
    assert (
        sc_low_ici.step_achievable_upper_us - sc_base.step_achievable_upper_us
    ) >= min_increase_us


def test_qwen35_tp16_outside_declared_8rank_group_is_hypothetical(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """TP16 outside declared 8-rank group makes whole_step hypothetical."""
    dep = advisor.DEPLOYMENT_PRESETS["qwen3.5-397b"]
    declaration = {
        "source": "unit test",
        "scopes": ["whole_step"],
        "ranks": 8,
    }
    declared = dataclasses.replace(
        dep, decode=dataclasses.replace(dep.decode, megakernel=declaration)
    )
    monkeypatch.setattr(support, "DEPLOYMENT_PRESETS", {"qwen3.5-397b": declared})
    prof = advisor.resolve_model_profile("qwen3.5-397b")
    r = advisor.advise(
        prof, advisor.WorkloadSpec(phase="decode", batch=16, tp=16), "tpu7x"
    )
    assert r.recommended_scope == "per_op"
    assert r.fastest_scope == "whole_step"
    ws = next(s for s in r.scopes if s.scope == "whole_step")
    assert "rank_count_outside_core_grid" in ws.blockers
    assert ws.support == "hypothetical"


def test_quantization_ridge_tokens_and_tp_invariance() -> None:
    """Verify W8A16/W8A8/BF16 ridge tokens on tpu7x/v6e and TP invariance."""
    c7x = advisor.chips.resolve_chip_spec("tpu7x")
    cv6e = advisor.chips.resolve_chip_spec("v6e")

    expected_ridge_w8a16_7x = 312.16
    expected_tokens_w8a16_7x = 157
    expected_ridge_w8a8_7x = 621.62
    expected_tokens_w8a8_7x = (311, 312, 313)
    expected_tokens_bf16_7x = 313
    expected_tokens_w8a16_v6 = 281
    expected_tokens_bf16_v6 = 561

    r_w8a16_7x = c7x.peak_flops("fp8", "bf16") / c7x.hbm_bw_bytes_per_s
    assert round(r_w8a16_7x, 2) == expected_ridge_w8a16_7x
    assert math.ceil(r_w8a16_7x / 2.0) == expected_tokens_w8a16_7x

    r_w8a8_7x = c7x.peak_flops("fp8", "fp8") / c7x.hbm_bw_bytes_per_s
    assert round(r_w8a8_7x, 2) == expected_ridge_w8a8_7x
    assert math.ceil(r_w8a8_7x / 2.0) in expected_tokens_w8a8_7x

    r_bf16_7x = c7x.peak_flops("bf16", "bf16") / c7x.hbm_bw_bytes_per_s
    assert math.ceil(r_bf16_7x / 1.0) == expected_tokens_bf16_7x

    r_w8a16_v6 = cv6e.peak_flops("fp8", "bf16") / cv6e.hbm_bw_bytes_per_s
    assert math.ceil(r_w8a16_v6 / 2.0) == expected_tokens_w8a16_v6

    r_bf16_v6 = cv6e.peak_flops("bf16", "bf16") / cv6e.hbm_bw_bytes_per_s
    assert math.ceil(r_bf16_v6 / 1.0) == expected_tokens_bf16_v6

    cfg_divisible = {
        "model_type": "llama",
        "num_hidden_layers": 8,
        "hidden_size": 4096,
        "intermediate_size": 11008,
        "num_attention_heads": 32,
        "num_key_value_heads": 16,
        "torch_dtype": "bfloat16",
        "quantization_config": {"quant_method": "fp8"},
    }
    prof_div = advisor.models.model_profile_from_hf_config(cfg_divisible)
    wl8 = advisor.specs.WorkloadSpec(phase="decode", batch=16, tp=8, context=0)
    wl16 = advisor.specs.WorkloadSpec(phase="decode", batch=16, tp=16, context=0)
    rep8 = advisor.advise(prof_div, wl8, c7x)
    rep16 = advisor.advise(prof_div, wl16, c7x)
    assert rep8.ridge_tokens == rep16.ridge_tokens


def test_mixed_precision_attention_qk_pv_executes_at_bf16_peak() -> None:
    """Verify mixed-precision attention dot products execute at bf16_peak."""
    c7x = advisor.chips.resolve_chip_spec("tpu7x")
    wl_prefill = advisor.specs.WorkloadSpec(
        phase="prefill", batch=1, isl=2048, osl=0, tp=8
    )
    prof_mixed = advisor.models.resolve_model_profile(
        "qwen3.5-397b", weight_dtype="fp8", act_dtype="bf16", kv_dtype="fp8"
    )
    ctx_mixed = _build_scope_context(prof_mixed, wl_prefill, c7x)
    assert ctx_mixed.attn_is_bf16 is True

    t_matmul, t_matmul_upper, _, _, _ = _compute_scope_matmuls(
        ctx_mixed, prof_mixed, wl_prefill, (1.0, 0.0, False)
    )
    expected_dense_matmul = (
        ctx_mixed.flops_dense - ctx_mixed.flops_kv
    ) / ctx_mixed.peak_w + ctx_mixed.flops_kv / ctx_mixed.bf16_peak
    t_moe = ctx_mixed.flops_moe / ctx_mixed.peak_e
    assert t_matmul == pytest.approx(expected_dense_matmul + t_moe)
    assert t_matmul_upper == pytest.approx(t_matmul)


def test_subchannel_quantization_vpu_bottleneck() -> None:
    """Verify subchannel block_k < 512 adds VPU ops and shifts compute bound."""
    c7x = advisor.chips.resolve_chip_spec("tpu7x")
    wl_prefill = advisor.specs.WorkloadSpec(
        phase="prefill", batch=1, isl=2048, osl=0, tp=8
    )
    cfg_w128 = {
        "model_type": "qwen2",
        "num_hidden_layers": 4,
        "hidden_size": 2048,
        "intermediate_size": 4096,
        "num_attention_heads": 16,
        "num_key_value_heads": 16,
        "quantization_config": {
            "quant_method": "fp8",
            "weight_block_size": [128, 128],
            "activation_scheme": "none",
        },
    }
    prof_w128 = advisor.models.model_profile_from_hf_config(cfg_w128)
    expected_block_k = 128
    assert prof_w128.weight_block_k == expected_block_k
    ctx_w128 = _build_scope_context(prof_w128, wl_prefill, c7x)
    _, _, v_dense_128, _, _ = _compute_scope_matmuls(
        ctx_w128, prof_w128, wl_prefill, (1.0, 0.0, False)
    )
    expected_subchannel_ops = (
        constants.SUBCHANNEL_VPU_OPS_PER_BLOCK
        * ctx_w128.active_tokens
        * ctx_w128.base_dense_params
        / expected_block_k
    )
    assert v_dense_128 >= expected_subchannel_ops
