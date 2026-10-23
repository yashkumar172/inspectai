"""Copy the precomputed files a hosted, dataset-free app needs into the repository's `hosted/` folder.

Usage (PowerShell):
    python scripts/export_hosted_bundle.py
    python scripts/export_hosted_bundle.py --categories cashew pcb1 --force

Reads <INSPECTAI_ARTIFACT_DIR> and <INSPECTAI_REPORT_DIR> (or the folders under
INSPECTAI_DATA_ROOT) and writes hosted/artifacts and hosted/reports: the evaluation results and
comparison figures for the dashboard, the Model C (PatchCore-inspired, dev profile) artifacts with
the ResNet-18 weights they need, and the classical-baseline artifacts. Model B, the full profile,
and per-run example figures are not exported. See inspectai.deployment.

Exit codes: 0 success; 2 configuration or data error; 3 artifact error; 1 any other failure.
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Sequence
from pathlib import Path

from inspectai.config import BUNDLE_ROOT, load_app_paths
from inspectai.deployment import HOSTED_MODEL_KEYS, export_hosted_bundle
from inspectai.errors import ArtifactError, ConfigError, InspectAIError

EXIT_OK = 0
EXIT_FAILURE = 1
EXIT_CONFIG_ERROR = 2
EXIT_ARTIFACT_ERROR = 3

logger = logging.getLogger("export_hosted_bundle")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--categories", nargs="+", help="categories to export (default: all found)")
    parser.add_argument(
        "--models",
        nargs="+",
        choices=HOSTED_MODEL_KEYS,
        default=["patchcore", "classical"],
        help="models whose artifacts are bundled (default: patchcore and classical)",
    )
    parser.add_argument("--output", type=Path, default=BUNDLE_ROOT, help="bundle folder")
    parser.add_argument("--force", action="store_true", help="replace an existing bundle")
    args = parser.parse_args(argv)
    try:
        paths = load_app_paths()
        summary = export_hosted_bundle(
            paths.artifact_dir,
            paths.report_dir,
            args.output,
            tuple(args.categories) if args.categories else None,
            args.force,
            tuple(args.models),
        )
    except ConfigError as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        return EXIT_CONFIG_ERROR
    except ArtifactError as exc:
        print(f"Artifact error: {exc}", file=sys.stderr)
        return EXIT_ARTIFACT_ERROR
    except InspectAIError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return EXIT_FAILURE
    print(
        f"Exported {summary.files} files ({summary.bytes / 1e6:.1f} MB) to {summary.destination}\n"
        f"  reports:   {', '.join(summary.report_categories) or 'none'}\n"
        + "".join(
            f"  artifacts {key}: {', '.join(cats) or 'none'}\n"
            for key, cats in summary.artifact_models.items()
        )
        + f"  weights:   {', '.join(summary.weights) or 'none'}"
    )
    return EXIT_OK


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    try:
        sys.exit(main())
    except Exception:
        logger.exception("Unexpected failure")
        sys.exit(EXIT_FAILURE)
