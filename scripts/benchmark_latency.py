"""Benchmark CPU inference latency of fitted models, with optional INT8 backbone variants.

Usage (PowerShell):
    python scripts/benchmark_latency.py --category cashew
    python scripts/benchmark_latency.py --category candle --models patchcore --variants fp32 static

For every (model, variant) pair a fresh worker process loads the saved artifact, warms up, times
`--runs` single-image `predict` calls (image decoding excluded, images pre-loaded), and scores the
whole test split once to get the image AUROC of that variant. A fresh process per pair keeps each
peak working set independent. Variants:

- `fp32`     the artifact as saved (the default inference path);
- `tiled`    patchcore only: fp32 with the tiled nearest-neighbor kernel (`--index-backend tiled`);
- `dynamic`  `torch.ao.quantization.quantize_dynamic` on the backbone. It converts only Linear
             layers, and the ResNet feature trunk has none, so it normally changes nothing; the
             report lists how many modules it converted so this is visible;
- `static`   post-training static INT8 of the ResNet-18 trunk, calibrated on normal fit images
             (the `dev` profile only).

The fitted tensors are never modified; only the feature extractor is swapped. Results go to
<INSPECTAI_REPORT_DIR>/benchmarks/latency_<category>.json and .md.

Exit codes: 0 success; 2 configuration or data error; 3 artifact error; 4 pretrained weights
unavailable; 1 any other failure.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import subprocess
import sys
import time
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import psutil
import torch

from inspectai.config import SplitConfig, load_paths
from inspectai.data.dataset import SUPPORTED_OBJECTS, VisADataset
from inspectai.data.splits import make_splits
from inspectai.errors import (
    ArtifactError,
    ConfigError,
    DatasetError,
    EvaluationError,
    ImageError,
    InspectAIError,
    ModelError,
    SplitError,
    WeightsNotFoundError,
)
from inspectai.evaluation.metrics import auroc
from inspectai.evaluation.runner import environment_info
from inspectai.inference.predictor import load_predictor, option_by_key
from inspectai.models.features import FeatureExtractor, images_to_batch, load_backbone
from inspectai.models.quantize import StaticInt8Extractor, dynamic_quantize

EXIT_OK = 0
EXIT_FAILURE = 1
EXIT_DATA_ERROR = 2
EXIT_ARTIFACT_ERROR = 3
EXIT_WEIGHTS_UNAVAILABLE = 4

MODELS = ("gaussian", "patchcore")
VARIANTS = ("fp32", "tiled", "dynamic", "static")
CALIBRATION_IMAGES = 64
RESULT_PREFIX = "RESULT "
logger = logging.getLogger("benchmark_latency")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Benchmark CPU inference latency.")
    parser.add_argument("--category", choices=SUPPORTED_OBJECTS, default="cashew")
    parser.add_argument("--models", nargs="+", choices=MODELS, default=list(MODELS))
    parser.add_argument("--variants", nargs="+", choices=VARIANTS, default=["fp32", "tiled"])
    parser.add_argument("--runs", type=int, default=100, help="timed predict calls (default 100)")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--artifact-dir", type=Path, help="Default: <INSPECTAI_ARTIFACT_DIR>.")
    parser.add_argument(
        "--output-dir", type=Path, help="Default: <INSPECTAI_REPORT_DIR>/benchmarks."
    )
    parser.add_argument("--worker", choices=MODELS, help=argparse.SUPPRESS)
    parser.add_argument("--variant", choices=VARIANTS, help=argparse.SUPPRESS)
    return parser


def percentile_summary(seconds: Sequence[float]) -> dict[str, float]:
    """Latency percentiles (ms) and throughput of a list of per-call times in seconds."""
    values = np.asarray(seconds, dtype=np.float64) * 1000.0
    if values.size == 0:
        raise EvaluationError("No timings to summarize")
    return {
        "p50_ms": round(float(np.percentile(values, 50)), 2),
        "p95_ms": round(float(np.percentile(values, 95)), 2),
        "p99_ms": round(float(np.percentile(values, 99)), 2),
        "mean_ms": round(float(values.mean()), 2),
        "max_ms": round(float(values.max()), 2),
        "throughput_images_per_s": round(float(1000.0 / values.mean()), 2),
    }


def _memory_gb() -> dict[str, float]:
    info = psutil.Process().memory_info()
    return {
        "working_set_gb": round(info.rss / 1024**3, 3),
        "peak_working_set_gb": round(int(getattr(info, "peak_wset", info.rss)) / 1024**3, 3),
    }


def _worker(args: argparse.Namespace) -> dict[str, object]:
    paths = load_paths()
    dataset = VisADataset.from_paths(paths)
    artifact_root = args.artifact_dir or paths.artifact_dir
    predictor = load_predictor(artifact_root, option_by_key(args.worker), args.category)
    model = predictor.model
    config = model.config  # type: ignore[attr-defined]
    splits = make_splits(dataset.samples, args.category, SplitConfig())
    by_rel = {s.image_rel: s for s in dataset.get_samples(args.category)}
    test = [by_rel[name] for name in sorted(splits.test)]
    images = [dataset.load_image(s) for s in test]
    labels = np.array([int(s.is_anomalous) for s in test])

    notes: dict[str, object] = {}
    if args.variant == "tiled":
        if args.worker != "patchcore":
            raise ConfigError("The tiled kernel applies to the patchcore model only")
        model.config = dataclasses.replace(config, index_backend="tiled")  # type: ignore[attr-defined]
    elif args.variant in ("dynamic", "static"):
        backbone, _ = load_backbone(config.backbone)
        if args.variant == "dynamic":
            quantized, report = dynamic_quantize(backbone, config.layers)
            notes["dynamic_quantization"] = report.to_dict()
            extractor = FeatureExtractor(quantized, config.layers, config.grid_layer)
        else:
            fit_names = sorted(splits.fit_files)[:CALIBRATION_IMAGES]
            calibration = [dataset.load_image(by_rel[name]) for name in fit_names]
            batches = [
                images_to_batch(calibration[i : i + 8], config.image_size, config.resize_mode)[0]
                for i in range(0, len(calibration), 8)
            ]
            extractor = StaticInt8Extractor(backbone, config.layers, config.grid_layer, batches)  # type: ignore[assignment]
            notes["static_quantization"] = {"calibration_images": len(calibration)}
        model.use_extractor(extractor)  # type: ignore[attr-defined]

    for index in range(args.warmup):
        model.predict(images[index % len(images)])
    timings: list[float] = []
    for index in range(args.runs):
        image = images[index % len(images)]
        start = time.perf_counter()
        model.predict(image)
        timings.append(time.perf_counter() - start)
    scores = np.array([model.predict(image).score for image in images])
    area = auroc(scores, labels)
    return {
        "model": args.worker,
        "variant": args.variant,
        "category": args.category,
        "runs": args.runs,
        "n_test_images": len(images),
        **percentile_summary(timings),
        **_memory_gb(),
        "image_auroc": None if area is None else round(float(area), 4),
        "num_threads": int(config.num_threads),  # type: ignore[attr-defined]
        **notes,
    }


def _run_worker_process(args: argparse.Namespace, model: str, variant: str) -> dict[str, object]:
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--worker",
        model,
        "--variant",
        variant,
        "--category",
        args.category,
        "--runs",
        str(args.runs),
        "--warmup",
        str(args.warmup),
    ]
    if args.artifact_dir:
        command += ["--artifact-dir", str(args.artifact_dir)]
    done = subprocess.run(command, capture_output=True, text=True, check=False)
    for line in reversed(done.stdout.splitlines()):
        if line.startswith(RESULT_PREFIX):
            return json.loads(line[len(RESULT_PREFIX) :])
    tail = (done.stderr or done.stdout).strip().splitlines()[-1:] or ["no output"]
    return {"model": model, "variant": variant, "error": tail[0], "exit_code": done.returncode}


def markdown_report(report: dict[str, object]) -> str:
    rows = report["results"]
    baseline = {
        r["model"]: r["image_auroc"]
        for r in rows  # type: ignore[union-attr]
        if r.get("variant") == "fp32" and "error" not in r
    }
    lines = [
        f"# Latency benchmark: {report['category']}",
        "",
        (
            f"Created {report['created_utc']}. {report['runs']} timed single-image `predict` calls per "
            "row after warm-up, image decoding excluded, one fresh process per row. Percentiles are "
            "linear interpolation over the timed calls (with 100 calls, p99 is close to the maximum)."
        ),
        "",
        (
            "| Model | Variant | p50 ms | p95 ms | p99 ms | Images/s | Peak working set | "
            "Image AUROC | vs fp32 |"
        ),
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:  # type: ignore[union-attr]
        if "error" in r:
            lines.append(f"| {r['model']} | {r['variant']} | failed: {r['error']} | | | | | | |")
            continue
        base = baseline.get(r["model"])
        delta = (
            "-" if base is None or r["image_auroc"] is None else f"{r['image_auroc'] - base:+.4f}"
        )
        lines.append(
            f"| {r['model']} | {r['variant']} | {r['p50_ms']:.1f} | {r['p95_ms']:.1f} | "
            f"{r['p99_ms']:.1f} | {r['throughput_images_per_s']:.2f} | "
            f"{r['peak_working_set_gb']:.2f} GB | {r['image_auroc']} | {delta} |"
        )
    dynamic = [r for r in rows if r.get("dynamic_quantization")]  # type: ignore[union-attr]
    if dynamic:
        d = dynamic[0]["dynamic_quantization"]
        lines += [
            "",
            (
                f"`dynamic`: {d['n_quantized_modules']} module(s) converted of "
                f"{d['n_conv_layers']} convolutions and {d['n_linear_layers']} linear layer(s); "
                f"covers the feature path: {d['covers_feature_path']}. Dynamic quantization has no "
                "convolution kernel, so it cannot speed up a ResNet feature trunk."
            ),
        ]
    env = report["environment"]
    lines += [
        "",
        (
            f"Machine: {env['processor']}, {env['physical_cores']} cores, {env['ram_total_gb']} GB, "
            f"power plan {env['power_plan']}, plugged in: {env['plugged_in']}; "
            f"PyTorch {torch.__version__}."
        ),
        "",
    ]
    return "\n".join(lines)


def _run(args: argparse.Namespace) -> int:
    if args.runs < 1 or args.warmup < 0:
        raise ConfigError("--runs must be at least 1 and --warmup at least 0")
    paths = load_paths()
    out_dir = args.output_dir or paths.report_dir / "benchmarks"
    results = []
    for model in args.models:
        for variant in args.variants:
            if variant == "tiled" and model != "patchcore":
                continue
            print(f"Benchmarking {model} / {variant} on {args.category} ...", flush=True)
            results.append(_run_worker_process(args, model, variant))
    report: dict[str, object] = {
        "category": args.category,
        "created_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "runs": args.runs,
        "warmup": args.warmup,
        "results": results,
        "environment": environment_info(),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = out_dir / f"latency_{args.category}"
    stem.with_suffix(".json").write_text(
        json.dumps(report, indent=1, allow_nan=False) + "\n", encoding="utf-8", newline="\n"
    )
    stem.with_suffix(".md").write_text(markdown_report(report), encoding="utf-8", newline="\n")
    print(markdown_report(report))
    print(f"Report written to {stem.with_suffix('.json').name} and .md in the benchmarks folder")
    return EXIT_FAILURE if any("error" in r for r in results) else EXIT_OK


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.worker:
            print(RESULT_PREFIX + json.dumps(_worker(args)))
            return EXIT_OK
        return _run(args)
    except WeightsNotFoundError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return EXIT_WEIGHTS_UNAVAILABLE
    except ArtifactError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return EXIT_ARTIFACT_ERROR
    except (ConfigError, DatasetError, SplitError, ImageError, EvaluationError, ModelError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return EXIT_DATA_ERROR
    except InspectAIError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return EXIT_FAILURE


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    try:
        sys.exit(main())
    except Exception:
        logger.exception("Unexpected failure")
        sys.exit(EXIT_FAILURE)
