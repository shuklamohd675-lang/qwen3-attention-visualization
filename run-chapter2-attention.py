#!/usr/bin/env python3
"""Run Chapter 2 experiment 2-2 locally, with pinned weights and saved evidence.

Examples (from the repository root)::
    .venv\\Scripts\\python.exe run-chapter2-attention.py --device cpu --quick
    .venv\\Scripts\\python.exe run-chapter2-attention.py --device cpu

The default full run follows the upstream protocol. --quick captures only the
Beijing prompt and is explicitly recorded as a partial experiment. No upstream
source, protocol, or validation/latest.json is modified.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
EXPERIMENT = ROOT / "chapter2" / "attention_visualization"
MODEL = "Qwen/Qwen3-0.6B"
REVISION = "c1899de289a04d12100db370d81485cdf75e47ca"


def progress(message: str) -> None:
    print(f"[{datetime.now(timezone.utc).strftime('%H:%M:%S UTC')}] {message}", flush=True)


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    result.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    result.add_argument("--threads", type=int, default=8, help="CPU thread limit (default: 8).")
    result.add_argument("--output-dir", type=Path, help="A new directory; defaults to .local/attention/<UTC timestamp>.")
    result.add_argument("--quick", action="store_true", help="Capture only the Beijing prompt; does not pass the full experiment.")
    result.add_argument("--max-new-tokens", type=int, default=None, help="Generation limit, 1..640; default uses the upstream protocol (640).")
    return result


def run(args: argparse.Namespace, output: Path, started: float) -> int:
    # Configure a non-interactive renderer before importing upstream plotting.
    os.environ.setdefault("MPLBACKEND", "Agg")
    os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".local" / "matplotlib"))
    os.environ.setdefault("HF_HOME", str(ROOT.parent / ".cache" / "huggingface"))
    sys.path.insert(0, str(EXPERIMENT))
    progress("Importing local model and plotting dependencies.")
    import matplotlib
    import matplotlib.pyplot as plt
    import numpy as np
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteria, StoppingCriteriaList
    from run_attention_experiment import capture, draw, matrix_metrics, region_indices, sha256
    from visualization import _configure_cjk_font

    _configure_cjk_font()
    raw_protocol = (EXPERIMENT / "attention_experiment_protocol.json").read_bytes()
    protocol = json.loads(raw_protocol)
    if protocol["model"] != MODEL or protocol["simple_prompt"] != "北京 的 天气 怎么样" or protocol["layers"] != [0, 13, -1]:
        raise ValueError("Upstream experiment protocol changed; inspect it before running this pinned reproduction.")
    (output / "experiment_protocol.json").write_bytes(raw_protocol)
    maximum = args.max_new_tokens if args.max_new_tokens is not None else protocol["max_new_tokens"]
    if not 1 <= maximum <= 640:
        raise ValueError("Generation limit must be between 1 and 640 tokens.")
    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false.")
    threads = min(args.threads, os.cpu_count() or args.threads)
    torch.set_num_threads(threads)
    try:
        torch.set_num_interop_threads(min(2, threads))
    except RuntimeError:
        pass
    dtype = torch.float32 if device == "cpu" else torch.float16
    row_tolerance = 2e-5 if device == "cpu" else 2e-3
    versions = {}
    for package in ("torch", "transformers", "numpy", "matplotlib", "huggingface-hub", "safetensors"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    parameters = {
        "model": MODEL, "revision": REVISION, "device": device, "dtype": str(dtype),
        "threads": threads, "attention_implementation": "eager", "mode": "quick" if args.quick else "full",
        "max_new_tokens": maximum, "seed": protocol["seed"], "do_sample": False,
        "layers": protocol["layers"], "row_sum_absolute_tolerance": row_tolerance,
        "causal_upper_triangle_tolerance": 1e-7,
        "generation_capture": "A full eager forward pass over the actually generated sequence.",
    }
    write_json(output / "run_parameters.json", parameters)
    progress(f"Loading cached {MODEL}, pinned revision {REVISION}, on {device} ({dtype}).")
    tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=REVISION, trust_remote_code=False, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, revision=REVISION, torch_dtype=dtype,
        attn_implementation="eager", trust_remote_code=False, local_files_only=True,
    ).to(device).eval()
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    revision = getattr(model.config, "_commit_hash", None)
    progress(f"Model loaded: {model.config.num_hidden_layers} layers; CPU thread limit {threads}.")

    def token_record(ids: list[int]) -> dict[str, Any]:
        return {
            "token_ids": ids,
            "tokens": [tokenizer.decode([item], skip_special_tokens=False) for item in ids],
            "tokenizer_pieces": tokenizer.convert_ids_to_tokens(ids),
            "token_note": "A token may contain only part of a UTF-8 character. Replacement glyphs in individual decoding are not whole words; IDs are authoritative.",
        }

    def checks(matrix: np.ndarray, expected_length: int) -> dict[str, Any]:
        finite = bool(np.isfinite(matrix).all())
        square = matrix.shape == (expected_length, expected_length)
        nonnegative = bool(finite and np.all(matrix >= 0))
        row_error = float(np.max(np.abs(matrix.sum(axis=1) - 1))) if finite else None
        upper_max = float(np.max(np.abs(np.triu(matrix, k=1)))) if finite else None
        return {
            "shape": list(matrix.shape), "expected_length": expected_length,
            "shape_matches_tokens": square, "finite": finite, "nonnegative": nonnegative,
            "max_row_sum_error": row_error, "row_sum_tolerance": row_tolerance,
            "rows_sum_to_one": bool(finite and row_error <= row_tolerance),
            "unmasked_upper_triangle_absolute_max": upper_max,
            "causal_triangle": bool(finite and upper_max <= 1e-7),
        }

    def measurements(matrices: dict[str, np.ndarray], length: int) -> dict[str, Any]:
        measured = {}
        for name, matrix in matrices.items():
            validation = checks(matrix, length)
            if not validation["finite"]:
                measured[name] = {"numerical_checks": validation}
                continue
            result = matrix_metrics(matrix)
            result["attention_sink_mean_excluding_first_row"] = float(matrix[1:, 0].mean()) if length > 1 else None
            result["attention_sink_note"] = "Mean including row 0 has a structural contribution of 1 from the first causal row."
            result["numerical_checks"] = validation
            measured[name] = result
        return measured

    def readable_plot(matrices: dict[str, np.ndarray], record: dict[str, Any]) -> None:
        labels = []
        for index, (tid, decoded) in enumerate(zip(record["token_ids"], record["tokens"])):
            piece = "[byte fragment]" if "\ufffd" in decoded else decoded.replace("\n", "\\n").replace("\t", "\\t")
            labels.append(f"{index}: {piece}\nID {tid}")
        fig, axes = plt.subplots(1, len(matrices), figsize=(19, 7.8), squeeze=False)
        for axis, (name, matrix) in zip(axes[0], matrices.items()):
            image = axis.imshow(matrix, origin="upper", cmap="viridis", vmin=0, vmax=1, aspect="equal")
            ticks = np.arange(len(labels))
            axis.set_xticks(ticks, labels, rotation=90, fontsize=7)
            axis.set_yticks(ticks, labels, fontsize=7)
            axis.set_xlabel("Key position (read from)")
            axis.set_ylabel("Query position (reading)")
            axis.set_title(f"{name}; average across heads\nsink without first row={matrix[1:, 0].mean():.1%}")
            fig.colorbar(image, ax=axis, fraction=0.046, pad=0.04, label="Attention weight")
        fig.suptitle("北京 的 天气 怎么样 | Real Qwen3-0.6B attention\nToken IDs identify byte fragments; these cells are not necessarily whole words.", fontsize=13)
        fig.tight_layout()
        fig.savefig(output / "beijing_tokens_readable.png", dpi=180, bbox_inches="tight")
        plt.close(fig)

    evidence: dict[str, Any] = {
        "experiment_id": "2-2", "status": "in_progress", "created_at": datetime.now(timezone.utc).isoformat(),
        "provider": "local Hugging Face Transformers", "model": MODEL,
        "requested_revision": REVISION, "model_revision": revision,
        "host": {"platform": platform.platform(), "machine": platform.machine(), "python": platform.python_version()},
        "dependencies": versions, "parameters": parameters,
        "source_hashes": {
            "local_runner": sha256(Path(__file__)),
            "upstream_runner": sha256(EXPERIMENT / "run_attention_experiment.py"),
            "upstream_visualization": sha256(EXPERIMENT / "visualization.py"),
        },
        "limitations": [
            "Head-averaged attention weights are not causal feature importance.",
            "Short-prompt observations do not establish general long-context position bias.",
            "Beginning/middle/end metrics follow the upstream positional statistic; magnitudes are outcomes, not success gates.",
        ],
    }
    progress("Capturing the exact Beijing prompt, layers 0, 13, and 27.")
    simple = tokenizer(protocol["simple_prompt"], return_tensors="pt", add_special_tokens=False)["input_ids"].to(device)
    simple_matrices, layers, heads = capture(model, simple, protocol["layers"])
    record = token_record(simple[0].tolist())
    evidence["architecture"] = {"layers": layers, "attention_heads": heads}
    evidence["simple"] = {"prompt": protocol["simple_prompt"], **record, "metrics": measurements(simple_matrices, len(record["token_ids"]))}
    arrays = {f"simple_{name}": matrix for name, matrix in simple_matrices.items()}
    np.savez_compressed(output / "attention_matrices.npz", **arrays)
    draw(simple_matrices, record["tokens"], output / "beijing_attention_layers.png", "Experiment 2-2: 北京 的 天气 怎么样")
    readable_plot(simple_matrices, record)
    write_json(output / "evidence.json", evidence)
    progress(f"Beijing prompt complete: {len(record['token_ids'])} tokens; matrices and two figures saved.")

    generated_matrices: dict[str, np.ndarray] = {}
    regions: dict[str, list[int]] = {"thinking": [], "answer": []}
    thinking_text = ""
    answer_text = ""
    if not args.quick:
        messages = [
            {"role": "system", "content": "你是一个会展示简短思考过程的助手。"},
            {"role": "user", "content": protocol["reasoning_prompt"]},
        ]
        rendered = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=True)
        inputs = tokenizer(rendered, return_tensors="pt", add_special_tokens=False)
        inputs = {key: value.to(device) for key, value in inputs.items()}
        context_length = int(inputs["input_ids"].shape[1])

        class CountProgress(StoppingCriteria):
            def __init__(self) -> None:
                self.last_count = 0
                self.last_time = time.monotonic()

            def __call__(self, input_ids, scores, **kwargs):
                count = int(input_ids.shape[1]) - context_length
                if count == 1 or count - self.last_count >= 32 or time.monotonic() - self.last_time >= 20:
                    progress(f"Generation progress: {count}/{maximum} new tokens (text retained only in artifacts).")
                    self.last_count, self.last_time = count, time.monotonic()
                return False

        torch.manual_seed(protocol["seed"])
        progress(f"Generating the upstream arithmetic task deterministically, up to {maximum} tokens.")
        with torch.no_grad():
            generated = model.generate(
                **inputs, max_new_tokens=maximum, do_sample=False,
                pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id,
                stopping_criteria=StoppingCriteriaList([CountProgress()]),
            )
        progress("Generation complete; capturing true attention over the prompt and generated sequence.")
        generated_matrices, _, _ = capture(model, generated, protocol["layers"])
        generated_ids = generated[0].tolist()
        generated_record = token_record(generated_ids)
        regions = region_indices(generated_record["tokens"], context_length)
        thinking_text = tokenizer.decode([generated_ids[i] for i in regions["thinking"]], skip_special_tokens=True)
        answer_text = tokenizer.decode([generated_ids[i] for i in regions["answer"]], skip_special_tokens=True).strip()
        completion = tokenizer.decode(generated[0, context_length:], skip_special_tokens=False)
        evidence["generated"] = {
            "prompt": protocol["reasoning_prompt"], "rendered_prompt": rendered,
            "context_length": context_length, "new_tokens": len(generated_ids) - context_length,
            **generated_record, "decoded_completion": completion, "regions": regions,
            "final_answer": answer_text,
            "termination": "eos" if generated_ids[-1] == tokenizer.eos_token_id else "token_limit_or_other",
            "metrics": measurements(generated_matrices, len(generated_ids)),
        }
        arrays.update({f"generated_{name}": matrix for name, matrix in generated_matrices.items()})
        np.savez_compressed(output / "attention_matrices.npz", **arrays)
        draw(generated_matrices, generated_record["tokens"], output / "reasoning_answer_attention_layers.png", "Experiment 2-2: generated reasoning and answer sequence")
        write_json(output / "evidence.json", evidence)
        progress("Generated-sequence matrices, region indices, and heatmap saved.")

    all_metrics = list(evidence["simple"]["metrics"].values()) + list(evidence.get("generated", {}).get("metrics", {}).values())
    numerical = [item["numerical_checks"] for item in all_metrics]
    content_without_markers = thinking_text.replace("<think>", "").replace("</think>", "").strip()
    gates = {
        "exact_model": model.config.model_type == "qwen3" and MODEL == protocol["model"],
        "pinned_real_model_revision": revision == REVISION,
        "beijing_prompt_exact": evidence["simple"]["prompt"] == "北京 的 天气 怎么样",
        "simple_three_layers_captured": set(simple_matrices) == {"layer_0", "layer_13", "layer_27"},
        "generated_three_layers_captured": set(generated_matrices) == {"layer_0", "layer_13", "layer_27"},
        "matrices_finite": all(item["finite"] for item in numerical),
        "matrices_nonnegative": all(item["nonnegative"] for item in numerical),
        "matrix_shapes_match_tokens": all(item["shape_matches_tokens"] for item in numerical),
        "rows_sum_to_one": all(item["rows_sum_to_one"] for item in numerical),
        "causal_triangle_exact": all(item["causal_triangle"] for item in numerical),
        "thinking_region_present": bool(regions["thinking"] and content_without_markers),
        "final_answer_region_present": bool(regions["answer"] and answer_text),
        "generation_completed_eos": evidence.get("generated", {}).get("termination") == "eos",
        "lossless_matrices_present": (output / "attention_matrices.npz").stat().st_size > 0,
        "heatmaps_present": all((output / name).is_file() and (output / name).stat().st_size > 0 for name in (
            "beijing_attention_layers.png", "beijing_tokens_readable.png", "reasoning_answer_attention_layers.png")),
    }
    simple_gate_names = [name for name in gates if name not in {"generated_three_layers_captured", "thinking_region_present", "final_answer_region_present", "generation_completed_eos", "heatmaps_present"}]
    evidence["simple_capture_passed"] = all(gates[name] for name in simple_gate_names)
    evidence["gates"] = gates
    evidence["status"] = "passed" if not args.quick and all(gates.values()) else "partial"
    evidence["elapsed_seconds"] = round(time.monotonic() - started, 3)
    evidence["completed_at"] = datetime.now(timezone.utc).isoformat()
    write_json(output / "evidence.json", evidence)
    summary = {
        "experiment_id": "2-2", "status": evidence["status"], "mode": parameters["mode"],
        "simple_capture_passed": evidence["simple_capture_passed"], "model": MODEL, "model_revision": revision,
        "device": device, "elapsed_seconds": evidence["elapsed_seconds"], "gates": gates,
        "simple_tokens": len(record["token_ids"]),
        "generated_tokens": evidence.get("generated", {}).get("new_tokens", 0),
        "sink_excluding_first_row": {name: value.get("attention_sink_mean_excluding_first_row") for name, value in evidence["simple"]["metrics"].items()},
        "note": "Quick mode is a partial experiment; full mode requires genuine thinking and final-answer regions. Generated thinking text is retained only in evidence.json.",
    }
    write_json(output / "summary.json", summary)
    manifest = {
        "experiment_id": "2-2", "status": evidence["status"], "gates": gates,
        "artifacts": {path.name: sha256(path) for path in sorted(output.iterdir()) if path.is_file()},
    }
    write_json(output / "manifest.json", manifest)
    progress(f"Finished: {evidence['status']}; simple capture passed={evidence['simple_capture_passed']}; evidence at {output}")
    if args.quick:
        return 0 if evidence["simple_capture_passed"] else 1
    return 0 if evidence["status"] == "passed" else 1


def main() -> int:
    arguments = parser()
    args = arguments.parse_args()
    if args.threads < 1:
        arguments.error("--threads must be >= 1")
    if args.max_new_tokens is not None and not 1 <= args.max_new_tokens <= 640:
        arguments.error("--max-new-tokens must be in 1..640")
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output = (args.output_dir if args.output_dir is not None else ROOT / ".local" / "attention" / timestamp).resolve()
    output.mkdir(parents=True, exist_ok=False)
    progress(f"New run directory: {output}")
    started = time.monotonic()
    try:
        return run(args, output, started)
    except (Exception, KeyboardInterrupt) as exc:
        failure = {
            "experiment_id": "2-2", "status": "partial", "error_type": type(exc).__name__,
            "error": str(exc), "elapsed_seconds": round(time.monotonic() - started, 3),
            "note": "Earlier artifacts, if any, are retained; this run did not complete full acceptance.",
        }
        write_json(output / "failure.json", failure)
        evidence_path = output / "evidence.json"
        if evidence_path.exists():
            previous = json.loads(evidence_path.read_text(encoding="utf-8"))
            previous["status"] = "partial"
            previous["failure"] = failure
            write_json(evidence_path, previous)
        progress(f"Run incomplete ({type(exc).__name__}): {exc}; details at {output / 'failure.json'}")
        return 130 if isinstance(exc, KeyboardInterrupt) else 1


if __name__ == "__main__":
    raise SystemExit(main())
