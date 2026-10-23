# SPDX-License-Identifier: Apache-2.0
"""Switches of the PP chunk scheduler that restore what the old trees ran.

Each defaults to this branch's behaviour:
- TPU_PP_INFLIGHT_COST=0: no in-flight step costing (4ec32d152 / 06f7b9bf9).
- TPU_SCHED_PAW_PROMOTE_WARMEST=0: FCFS admission, no warmest-first promotion.
"""

from types import SimpleNamespace
from unittest.mock import patch

from vllm.v1.core.sched.scheduler import Scheduler

from vllm_torchtpu.core.pp_chunk_scheduler import TpuPipelineChunkScheduler
from vllm_torchtpu.core.pp_chunks import StepCostModel


BUCKETS = [4096, 8192, 12288, 16384]
_EMPTY_STEP = SimpleNamespace(total_num_scheduled_tokens=0)


def _model(attn=1.5e-7, slack=0.1, granularity=4096) -> StepCostModel:
    linear = {b: 12.0 + 0.0109 * b for b in BUCKETS}
    full = linear[16384] + attn * 16384**2 / 2
    return StepCostModel(BUCKETS, linear, attn, full * (1 + slack), granularity)


def _sched(cost=None, running=(), waiting=(), **attrs) -> TpuPipelineChunkScheduler:
    s = TpuPipelineChunkScheduler.__new__(TpuPipelineChunkScheduler)
    s._cost = cost
    s._step_limit = None
    s._pp_size = 8
    s.waiting = list(waiting)
    s.vllm_config = SimpleNamespace(
        scheduler_config=SimpleNamespace(max_num_batched_tokens=16384),
        parallel_config=SimpleNamespace(pipeline_parallel_size=8),
    )
    s._mamba_split = False
    s.need_mamba_block_aligned_split = True
    s.running = list(running)
    s._step_tokens = 0
    s._step_pairs = 0.0
    s._step_squares = 0.0
    s._step_prefill_tokens = 0
    s._step_schedule = 0
    s._seeded = set()
    s._inflight_ms = []
    s._clips = 0
    s._waits = 0
    s._retention = None
    s.kv_cache_manager = SimpleNamespace()
    for k, v in attrs.items():
        setattr(s, k, v)
    return s


def _req(computed, prompt, rid="r", num_tokens=None):
    return SimpleNamespace(num_computed_tokens=computed, num_prompt_tokens=prompt,
                           num_tokens=prompt if num_tokens is None else num_tokens, request_id=rid)


def test_inflight_off_keeps_no_history():
    model = _model()
    est = model.step_ms(4096, 0.0, float(4096**2))
    on = _sched(cost=model)
    on._pp_size = 4
    on._enter(est)
    on.pipeline_advanced(pushes=2)
    assert on._inflight_ms == [0.0, 0.0, est]
    off = _sched(cost=model, _inflight_cost=False)
    off._pp_size = 4
    off._enter(est)
    off.pipeline_advanced(pushes=2)
    off.pipeline_advanced(settled=True)
    assert off._inflight_ms == []
    off.running = [_req(0, 8192)]
    with patch.object(Scheduler, "schedule", return_value=_EMPTY_STEP):
        off.schedule()
    assert off._inflight_ms == []




def test_chunk_scheduler_promotion_follows_the_switch():
    calls = []
    for on in (True, False):
        s = _sched(cost=None, _promote_waiting=on)
        s._select_waiting_queue_for_scheduling = lambda: calls.append(on)
        with patch.object(Scheduler, "schedule", return_value=_EMPTY_STEP):
            s.schedule()
    assert calls == [True]

