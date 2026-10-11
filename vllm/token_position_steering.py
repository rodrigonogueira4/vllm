# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Experimental eager Qwen3.5 residual steering by absolute input position.

SamplingParams.extra_args['token_position_steering'] contains layer, vector,
positions and coefficient. Positions refer to consumed tokens, not predictions.
This module intentionally has no imports from the vLLM runtime, so its packed
batch and residual contracts can be tested without CUDA.
"""

import math
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass

import torch

KEY = "token_position_steering"


@dataclass(frozen=True)
class SteeringSpec:
    layer: int
    vector: tuple[float, ...]
    positions: frozenset[int]
    coefficient: float

    @classmethod
    def parse(cls, value):
        if not isinstance(value, dict) or set(value) != {
            "layer",
            "vector",
            "positions",
            "coefficient",
        }:
            raise ValueError("Steering requires layer, vector, positions, coefficient")
        layer, positions = value["layer"], value["positions"]
        if type(layer) is not int or layer < 0:
            raise ValueError("Layer must be a nonnegative integer")
        if not isinstance(positions, list) or any(
            type(p) is not int or p < 0 for p in positions
        ):
            raise ValueError("Positions must be nonnegative integer token offsets")
        if len(positions) != len(set(positions)):
            raise ValueError("Duplicate positions are not allowed")
        vector = value["vector"]
        coefficient = value["coefficient"]
        if not isinstance(vector, list) or not vector:
            raise ValueError("Vector must be a nonempty list")
        if any(type(x) not in (int, float) or not math.isfinite(x) for x in vector):
            raise ValueError("Vector must contain finite real values")
        if type(coefficient) not in (int, float) or not math.isfinite(coefficient):
            raise ValueError("Coefficient must be finite")
        return cls(layer, tuple(vector), frozenset(positions), float(coefficient))


def validate_runtime(config, spec):
    """Reject modes that have not passed the steering conformance gate."""
    m, p, c, s = (
        config.model_config,
        config.parallel_config,
        config.cache_config,
        config.scheduler_config,
    )
    if m.hf_text_config.model_type not in ("qwen3_5_text", "qwen3_5_moe_text"):
        raise ValueError("Token steering currently supports Qwen3.5 text only")
    if spec.layer >= m.hf_text_config.num_hidden_layers:
        raise ValueError("Steering layer is outside the decoder")
    if len(spec.vector) != m.hf_text_config.hidden_size:
        raise ValueError("Steering vector width differs from hidden size")
    if not m.enforce_eager or c.enable_prefix_caching or s.async_scheduling:
        raise ValueError(
            "Steering requires eager, no prefix cache, synchronous scheduling"
        )
    if config.speculative_config is not None or config.kv_transfer_config is not None:
        raise ValueError("Steering does not support speculation or KV transfer")
    if any(
        getattr(p, name) != 1
        for name in (
            "tensor_parallel_size",
            "pipeline_parallel_size",
            "data_parallel_size",
        )
    ) or getattr(p, "enable_dbo", False):
        raise ValueError("Steering currently requires one worker and no microbatching")
    if getattr(config, "lora_config", None) is not None:
        raise ValueError("Steering conformance requires native weights without LoRA")


@dataclass(frozen=True)
class PackedSteering:
    spec: SteeringSpec
    rows: tuple[int, ...]


def pack_steering(request_ids, counts, offsets, specs):
    """Map each request's absolute mask into the current packed token rows."""
    if not (len(request_ids) == len(counts) == len(offsets)):
        raise ValueError("Request IDs, counts and offsets must have equal length")
    if len(set(request_ids)) != len(request_ids):
        raise ValueError("Request IDs must be unique in a packed batch")
    packed, row = [], 0
    for rid, count, offset in zip(request_ids, counts, offsets):
        if count < 0 or offset < 0:
            raise ValueError("Negative token counts or offsets")
        spec = specs.get(rid)
        if spec is not None and spec.coefficient != 0:
            rows = tuple(row + i for i in range(count) if offset + i in spec.positions)
            if rows:
                packed.append(PackedSteering(spec, rows))
        row += count
    return tuple(packed)


_active: ContextVar[tuple[PackedSteering, ...]] = ContextVar("steering", default=())


@contextmanager
def steering_context(packed):
    token = _active.set(packed)
    try:
        yield
    finally:
        _active.reset(token)


def apply_steering(layer, hidden_states, residual):
    """Add to the materialized block output once, preserving all unmasked rows.

    vLLM defers the FFN residual addition to the following RMSNorm. For selected
    rows only, materialize that sum in model dtype, add the cast delta, and zero
    its residual slot so the next norm cannot add the residual a second time.
    """
    entries = [p for p in _active.get() if p.spec.layer == layer]
    if not entries:
        return hidden_states, residual
    h = hidden_states.clone()
    r = residual.clone() if residual is not None else None
    for entry in entries:
        spec = entry.spec
        if len(spec.vector) != h.shape[-1] or max(entry.rows) >= h.shape[0]:
            raise ValueError("Steering shape or packed row mismatch")
        rows = torch.tensor(entry.rows, dtype=torch.long, device=h.device)
        delta = (
            torch.tensor(spec.vector, device=h.device, dtype=torch.float32)
            * spec.coefficient
        ).to(h.dtype)
        combined = h[rows] if r is None else h[rows] + r[rows]
        h[rows] = combined + delta
        if r is not None:
            r[rows] = 0
    return h, r


def runner_context(runner, scheduler_output):
    """Read the same request order and computed-token offsets as input packing."""
    ids = runner.input_batch.req_ids
    specs = {}
    for rid in ids:
        request = runner.requests[rid]
        params = request.sampling_params
        raw = (params.extra_args or {}).get(KEY) if params is not None else None
        if raw is not None:
            spec = SteeringSpec.parse(raw)
            validate_runtime(runner.vllm_config, spec)
            if request.mm_features or request.prompt_embeds is not None:
                raise ValueError("Steering currently accepts text token IDs only")
            specs[rid] = spec
    counts = [scheduler_output.num_scheduled_tokens[rid] for rid in ids]
    offsets = [int(x) for x in runner.input_batch.num_computed_tokens_cpu[: len(ids)]]
    return steering_context(pack_steering(ids, counts, offsets, specs))


def validate_new_requests(config, requests):
    """Validate before a new request can consume cached state on this worker."""
    for request in requests:
        params = request.sampling_params
        raw = (params.extra_args or {}).get(KEY) if params is not None else None
        if raw is not None:
            validate_runtime(config, SteeringSpec.parse(raw))
            if request.mm_features or request.prompt_embeds is not None:
                raise ValueError("Steering currently accepts text token IDs only")
