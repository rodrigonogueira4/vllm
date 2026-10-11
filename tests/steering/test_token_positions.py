# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU contract tests: no CUDA runtime or model weights are loaded."""

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace as NS

import pytest
import torch

path = Path(__file__).resolve().parents[2] / "vllm/token_position_steering.py"
loader = importlib.util.spec_from_file_location("steering_contract", path)
assert loader is not None and loader.loader is not None
steering = importlib.util.module_from_spec(loader)
sys.modules[loader.name] = steering
loader.loader.exec_module(steering)


def spec(positions, coefficient=1, layer=2, vector=None):
    return steering.SteeringSpec.parse(
        dict(
            layer=layer,
            positions=positions,
            coefficient=coefficient,
            vector=vector if vector is not None else [0.25, -0.5],
        )
    )


def config():
    return NS(
        model_config=NS(
            enforce_eager=True,
            hf_text_config=NS(
                model_type="qwen3_5_moe_text", num_hidden_layers=40, hidden_size=2
            ),
        ),
        parallel_config=NS(
            tensor_parallel_size=1, pipeline_parallel_size=1, data_parallel_size=1
        ),
        cache_config=NS(enable_prefix_caching=False),
        scheduler_config=NS(async_scheduling=False),
        speculative_config=None,
        kv_transfer_config=None,
    )


def test_chunked_prefill_decode_and_batch_reordering_use_absolute_positions():
    a, b = spec([1, 4, 8]), spec([0, 5], vector=[2, 3])
    p = steering.pack_steering(
        ["b", "a", "off"], [2, 3, 2], [5, 3, 0], {"a": a, "b": b}
    )
    assert [(x.spec, x.rows) for x in p] == [(b, (0,)), (a, (3,))]
    p = steering.pack_steering(["a"], [1], [8], {"a": a})
    assert p[0].rows == (0,)
    assert not steering.pack_steering(["a"], [1], [9], {"a": a})


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_masked_residual_matches_materialized_block_and_preserves_other_rows(dtype):
    h = torch.tensor([[1, 2], [3, 4], [5, 6], [7, 8]], dtype=dtype)
    r = torch.tensor([[0.5, 1], [1, 2], [1.5, 3], [2, 4]], dtype=dtype)
    expected = (h[1] + r[1]) + torch.tensor([0.25, -0.5], dtype=dtype)
    packed = steering.pack_steering(["a"], [4], [0], {"a": spec([1])})
    with steering.steering_context(packed):
        out, residual = steering.apply_steering(2, h, r)
    assert torch.equal(out[1] + residual[1], expected)
    assert torch.equal(out[[0, 2, 3]], h[[0, 2, 3]])
    assert torch.equal(residual[[0, 2, 3]], r[[0, 2, 3]])
    assert torch.equal(h[1], torch.tensor([3, 4], dtype=dtype))


def test_zero_dose_wrong_layer_and_context_cleanup_are_exact_identity():
    h, r = torch.randn(3, 2), torch.randn(3, 2)
    zero = steering.pack_steering(["a"], [3], [0], {"a": spec([0], 0)})
    with steering.steering_context(zero):
        a, b = steering.apply_steering(2, h, r)
        assert a is h and b is r
    packed = steering.pack_steering(["a"], [3], [0], {"a": spec([0])})
    with pytest.raises(RuntimeError), steering.steering_context(packed):
        a, b = steering.apply_steering(1, h, r)
        assert a is h and b is r
        raise RuntimeError("simulate failed forward")
    a, b = steering.apply_steering(2, h, r)
    assert a is h and b is r


def test_mixed_vectors_and_padding_do_not_leak_between_requests():
    packed = steering.pack_steering(
        ["a", "b", "off"], [2, 1, 1], [0, 10, 0], {"a": spec([0]), "b": spec([10], -2)}
    )
    h = torch.zeros(6, 2)
    with steering.steering_context(packed):
        out, r = steering.apply_steering(2, h, None)
    assert r is None
    assert torch.equal(
        out, torch.tensor([[0.25, -0.5], [0, 0], [-0.5, 1], [0, 0], [0, 0], [0, 0]])
    )


@pytest.mark.parametrize(
    "change",
    [
        {"positions": [-1]},
        {"positions": [True]},
        {"positions": [2, 2]},
        {"vector": [float("nan")]},
        {"coefficient": float("inf")},
        {"layer": -1},
    ],
)
def test_malformed_interventions_fail_closed(change):
    raw = dict(layer=2, positions=[0], coefficient=1, vector=[0.25, -0.5])
    with pytest.raises(ValueError):
        steering.SteeringSpec.parse(raw | change)


@pytest.mark.parametrize(
    "section, key, value",
    [
        ("model_config", "enforce_eager", False),
        ("cache_config", "enable_prefix_caching", True),
        ("scheduler_config", "async_scheduling", True),
        ("parallel_config", "tensor_parallel_size", 2),
    ],
)
def test_unvalidated_execution_modes_fail_closed(section, key, value):
    c = config()
    steering.validate_runtime(c, spec([0]))
    setattr(getattr(c, section), key, value)
    with pytest.raises(ValueError):
        steering.validate_runtime(c, spec([0]))


def test_runner_uses_each_requests_sampling_parameters_and_current_offsets():
    raw = dict(layer=2, positions=[7], coefficient=1, vector=[0.25, -0.5])
    runner = NS(
        vllm_config=config(),
        input_batch=NS(req_ids=["b", "a"], num_computed_tokens_cpu=[7, 0]),
        requests={
            "b": NS(
                sampling_params=NS(extra_args={steering.KEY: raw}),
                mm_features=[],
                prompt_embeds=None,
            ),
            "a": NS(
                sampling_params=NS(extra_args={}), mm_features=[], prompt_embeds=None
            ),
        },
    )
    h = torch.zeros(3, 2)
    with steering.runner_context(runner, NS(num_scheduled_tokens={"a": 2, "b": 1})):
        out, _ = steering.apply_steering(2, h, None)
    assert torch.equal(out, torch.tensor([[0.25, -0.5], [0, 0], [0, 0]]))
