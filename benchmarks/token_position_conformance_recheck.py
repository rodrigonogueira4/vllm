# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Rerun frozen steering conformance in a new directory.

The caller supplies an existing, hash-frozen experiment and its HF reference
outputs. Change one numerical implementation option; all cases, masks,
thresholds, and conditional performance checks remain in the frozen probe.
"""

import argparse
import hashlib
import importlib
import json
import os
import shutil
import sys
import time
from pathlib import Path


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--mode",
        choices=["batch-invariant", "hf-residual-norm", "gdn-triton"],
        required=True,
    )
    args = parser.parse_args()
    root = args.experiment.resolve()
    output = args.output.resolve()
    invariant = args.mode == "batch-invariant"
    assert os.environ.get("VLLM_BATCH_INVARIANT", "0") == str(int(invariant))
    if args.mode == "gdn-triton":
        assert os.environ.get("VLLM_GDN_DECODE_KERNEL") == "triton"
    assert not output.exists(), "Use a fresh directory; never append another trial."
    manifest = json.loads((root / "source_manifest.json").read_text())
    for name, expected in manifest.items():
        assert digest(root / name) == expected, name
    original = root / "results"
    references = sorted(original.glob("hf_*"))
    assert references and all(path.is_file() for path in references)
    hashes = {path.name: digest(path) for path in references}
    output.mkdir(parents=True)
    for path in references:
        shutil.copyfile(path, output / path.name)
        assert digest(output / path.name) == hashes[path.name]
    (output / "recheck_inputs.json").write_text(
        json.dumps(
            {
                "at": time.time(),
                "experiment": str(root),
                "source_manifest_sha256": digest(root / "source_manifest.json"),
                "reference_hashes": hashes,
                "script_sha256": digest(Path(__file__)),
                "diagnostic_mode": args.mode,
                "environment": {
                    "VLLM_BATCH_INVARIANT": str(int(invariant)),
                    "VLLM_GDN_DECODE_KERNEL": os.environ.get(
                        "VLLM_GDN_DECODE_KERNEL", "cuda"
                    ),
                },
                "cases_and_gates_unchanged": True,
                "scientific_adoption": False,
            },
            indent=2,
        )
        + "\n"
    )
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(root))
    common = importlib.import_module("common")
    common.OUT = output
    common.CFG = dict(common.CFG)
    common.CFG["fork_commit"] = os.environ["VLLM_DIAGNOSTIC_FORK_COMMIT"]
    if args.mode in ("hf-residual-norm", "gdn-triton"):
        import vllm

        original_llm = vllm.LLM

        def configured_llm(**kwargs):
            extra = dict(kwargs["additional_config"])
            if args.mode == "hf-residual-norm":
                extra["token_position_steering_hf_norm"] = True
            else:
                extra["gdn_prefill_backend"] = "triton"
            return original_llm(**(kwargs | {"additional_config": extra}))

        vllm.LLM = configured_llm

    def stage(name: str) -> None:
        (output / "current-stage.txt").write_text(name + "\n")
        common.save("progress.json", {"stage": name, "at": time.time()})

    common.stage = stage
    probe = importlib.import_module("fork_probe")
    probe.main()
    for name, expected in hashes.items():
        assert digest(original / name) == expected, name
    stage("recheck_complete")


if __name__ == "__main__":
    main()
