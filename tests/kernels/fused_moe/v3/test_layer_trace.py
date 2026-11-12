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
"""fused_ep_moe traced on the CPU, on an abstract mesh of 1 to 8 shards.

The layer has to trace on the installed JAX (the repo pins jax==0.10.2): the
transport and FFN kernels issue their DMAs through `kernels.pallas_compat`,
because only an internal Pallas build takes `priority` / `strict_ordering` on
`make_async_copy` and `make_async_remote_copy`. The expert weight refills
keep their queue (Config.weight_dma_priority) and every other DMA stays on
queue 0.
"""
import contextlib
import functools
from unittest import mock

import jax
import jax.numpy as jnp
import pytest
from jax._src.pallas.mosaic import tpu_info

from vllm_torchtpu.kernels.fused_moe import v3 as fused_ep
from vllm_torchtpu.kernels.fused_moe.v3 import (Config, ffn_kernel, layer,
                                                layout)
from vllm_torchtpu.kernels.fused_moe.v3.rowquant import FP8

pytestmark = pytest.mark.cpu_test

EXPERTS, HIDDEN, INTER, TOKENS, TOP_K, TILE_ROWS = 128, 1024, 512, 2048, 8, 128
WEIGHT_SHAPES = ((HIDDEN, 2 * INTER), (INTER, HIDDEN))


@pytest.fixture(autouse=True)
def _on_the_host():
    with jax.default_device(jax.devices("cpu")[0]):
        yield


@contextlib.contextmanager
def _tpu_info_on_cpu():
    """The layer asks for the TPU generation; answer TPU7x for the CPU
    device. The layer's program caches and JAX's are cleared so every call
    traces the kernels afresh."""
    tpu_info.get_tpu_info.cache_clear()
    jax.clear_caches()
    try:
        with mock.patch.dict(
                tpu_info.registry, {
                    "cpu":
                    lambda: tpu_info.get_tpu_info_for_chip(
                        tpu_info.ChipVersion.TPU_7X, 1)
                }), mock.patch.dict(layer._LAYER_CACHE,
                                    clear=True), mock.patch.dict(
                                        ffn_kernel._BUILD_CACHE, clear=True):
            yield
    finally:
        tpu_info.get_tpu_info.cache_clear()
        jax.clear_caches()


def _trace_layer(num_shards, config=None):
    mesh = jax.sharding.AbstractMesh((num_shards, ), (layout.MESH_AXIS, ))
    layer_fn = functools.partial(fused_ep.fused_ep_moe,
                                 top_k=TOP_K,
                                 renormalize=True,
                                 mesh=mesh,
                                 tile_rows=TILE_ROWS,
                                 config=config)
    sd = jax.ShapeDtypeStruct
    with _tpu_info_on_cpu():
        return jax.make_jaxpr(layer_fn)(
            sd((TOKENS, HIDDEN), jnp.bfloat16),
            sd((EXPERTS, HIDDEN, 2 * INTER), FP8),
            sd((EXPERTS, INTER, HIDDEN), FP8),
            sd((EXPERTS, 2 * INTER), jnp.float32),
            sd((EXPERTS, HIDDEN), jnp.float32),
            sd((TOKENS, EXPERTS), jnp.float32),
        )


def _dma_priorities(jaxpr):
    """(the priorities of the expert weight refills, those of every other
    `dma_start`) over `jaxpr` and its sub-jaxprs."""
    weights, others = [], []

    def walk(value):
        if isinstance(value, (tuple, list)):
            for item in value:
                walk(item)
        elif hasattr(value, "eqns"):
            for eqn in value.eqns:
                if eqn.primitive.name == "dma_start":
                    src, dst, *_ = eqn.params["tree"].unflatten(eqn.invars)
                    src = getattr(src, "ref", src).aval
                    dst = getattr(dst, "ref", dst).aval
                    is_weight = (str(src.memory_space) == "hbm"
                                 and str(dst.memory_space) == "vmem"
                                 and dst.dtype == FP8
                                 and dst.shape[-2:] in WEIGHT_SHAPES)
                    (weights if is_weight else others).append(
                        eqn.params["priority"])
                walk(list(eqn.params.values()))
        elif hasattr(value, "jaxpr"):
            walk(value.jaxpr)

    walk(jaxpr)
    return weights, others


@pytest.mark.parametrize("sorted_rows", [False, True])
@pytest.mark.parametrize("weight_dma_priority", [0, 1])
@pytest.mark.parametrize("num_shards", [1, 2, 8])
def test_the_weight_refills_keep_their_queue(num_shards, weight_dma_priority,
                                             sorted_rows):
    jaxpr = _trace_layer(
        num_shards,
        Config(weight_dma_priority=weight_dma_priority,
               sorted_rows=sorted_rows))
    weights, others = _dma_priorities(jaxpr.jaxpr)
    # w1 and w2 of the first visits, and their refills.
    assert len(weights) >= 4
    assert set(weights) == {weight_dma_priority}
    assert others and set(others) == {0}
