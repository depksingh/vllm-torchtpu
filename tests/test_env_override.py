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
"""The SparseCore collective-offload flags env_override adds to LIBTPU_INIT_ARGS.

The offload thresholds follow the chip family, read from TPU_ACCELERATOR_TYPE
or the GCE metadata server; a flag in LIBTPU_INIT_ARGS wins.
"""
import importlib
import os
import sys
from types import SimpleNamespace
from unittest import mock

import pytest

import vllm_torchtpu.env_override as env_override

pytestmark = pytest.mark.cpu_test

_AR_OFFLOAD = "--xla_tpu_enable_sparse_core_collective_offload_all_reduce"
_AR_MIN = "--xla_tpu_sparse_core_all_reduce_offload_min_size_in_bytes"
_AG_MIN = "--xla_tpu_sparse_core_all_gather_offload_min_size_in_bytes"
_MIB = 1 << 20


@pytest.fixture
def libtpu_flags(monkeypatch):
    """flags(env, metadata_type=None): the LIBTPU_INIT_ARGS env_override
    assembles from `env` alone, as a list of (flag, value). The metadata
    server answers `metadata_type` (None: not found)."""

    def flags(env, metadata_type=None):

        def get(*args, **kwargs):
            return SimpleNamespace(status_code=200 if metadata_type else 404,
                                   text=metadata_type or "")

        monkeypatch.setitem(sys.modules, "requests", SimpleNamespace(get=get))
        with mock.patch.dict(os.environ, env, clear=True):
            importlib.reload(env_override)
            args = os.environ.get("LIBTPU_INIT_ARGS", "")
        return [tuple(token.split("=", 1)) for token in args.split()]

    return flags


def _sc_flags(flags):
    return {
        name: value
        for name, value in flags if name in (_AR_OFFLOAD, _AR_MIN, _AG_MIN)
    }


@pytest.mark.parametrize("env, metadata_type", [
    ({
        "TPU_ACCELERATOR_TYPE": "tpu7x-8"
    }, None),
    ({}, "tpu7x-8"),
    ({
        "TPU_ACCELERATOR_TYPE": "v7x"
    }, None),
])
def test_tpu7x_auto_threshold_turns_sc_all_reduce_offload_off(
        libtpu_flags, env, metadata_type):
    assert _sc_flags(libtpu_flags(env, metadata_type)) == {
        _AR_OFFLOAD: "false",
        _AR_MIN: str(64 * _MIB),
        _AG_MIN: str(64 * _MIB),
    }


@pytest.mark.parametrize("value", ["0", "not-a-number"])
def test_disabled_threshold_adds_no_sc_flags(libtpu_flags, value):
    flags = libtpu_flags({
        "TPU_ACCELERATOR_TYPE": "tpu7x-8",
        "SC_ALLREDUCE_ALLGATHER_OFFLOAD_MIN_BYTES": value,
    })
    assert _sc_flags(flags) == {}


def test_user_flag_wins(libtpu_flags):
    flags = libtpu_flags({
        "TPU_ACCELERATOR_TYPE": "tpu7x-8",
        "LIBTPU_INIT_ARGS": f"{_AR_OFFLOAD}=true",
    })
    assert [value for name, value in flags if name == _AR_OFFLOAD] == ["true"]


def test_chip_family_comes_only_from_tpu_type_or_metadata(libtpu_flags):
    """Other variables that name a chip (such as a benchmark harness's
    target) are not read."""
    flags = libtpu_flags({"HIEREVO_TARGET_NAME": "tpu7x"})
    assert _sc_flags(flags) == {}
