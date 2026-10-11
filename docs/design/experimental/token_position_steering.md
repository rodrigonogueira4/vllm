# Experimental token-position steering

Fork branch `token-position-steering-20261011`, based on v0.30.0
`ced6857afa0ea7b2e3f0846a62e1394e90f15607`. This is new engineering work;
the previous vLLM-lens validation remains closed and failed.

For text-only Qwen3.5, send one intervention per request through the existing
serialized sampling parameters:

```python
SamplingParams(extra_args={"token_position_steering": {
    "layer": 24, "vector": vector_float32.tolist(),
    "positions": original_task_token_positions, "coefficient": 1.0,
}})
```

Positions are zero-based absolute **consumed input** token positions. They are
not next-token predictions or local packed-batch offsets. Prefill chunks,
decode, request reordering and mixed off/on requests all use the runner's
request order and computed-token offsets. Omitting future positions prevents
injection into generated tokens; earlier interventions can still affect caches.
Do not interpret mask exclusion as absence of downstream causal effects.

The inherited Qwen3Next/Qwen3.5 decoder loop exposes the FFN and residual
separately. At selected rows, materialize their sum in model dtype, add the
FP32-scaled/BF16-cast vector, then set that row's residual to zero. This targets
the complete decoder output before final norm without double-adding the residual.
Unselected rows retain both original components byte for byte; zero coefficient
is an exact no-op. Context is cleared even when model execution raises.

Initial supported execution: eager, synchronous scheduling, one GPU, no prefix
caching, no KV transfer, no speculative decode, no LoRA, no multimodal inputs.
These checks fail closed. CUDA graphs, multi-worker and cache-key extensions
are future work, not validated features. Masks are carried in SamplingParams;
no separate global per-batch vector is shared between requests.

## Validation status

The CPU unit contract covers packed mapping, chunked inputs, decode, BF16
residual arithmetic, mixed vectors, padding, zero dose, cleanup and unsupported
modes. Run without importing the CUDA fixture tree:

```bash
.venv/bin/python -m pytest --confcutdir=tests/steering tests/steering -q
```

These tests are **not a model equivalence result**. A native Qwen3.5 HF/vLLM
comparison on GPU is still required before any scientific collection: save raw
activations/logits/token IDs and masks; native zero/cleanup gate 1e-6; exactly
zero off-mask changes; selected projection within 6%; original cross-backend
probability tolerance 0.03 and 100% action argmax agreement; cached teacher
forcing, sampler/EOS and matched-work timing. Do not relax a gate to adopt this
backend. The prior 15.48 percentage-point gap is not explained by this patch.
Current HF experiments remain unchanged.

AI-assisted implementation by Codex. No upstream PR or human model-validation
approval is asserted.
