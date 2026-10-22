"""Download pretrained backbone weights into the torchvision cache (an explicit action).

Usage (PowerShell):
    python scripts/fetch_weights.py --backbone resnet18
    python scripts/fetch_weights.py --backbone wide_resnet50_2

The weights are stored under <TORCH_HOME>\\hub\\checkpoints and verified against the hash in
their file name. Nothing else in this project downloads weights unless
INSPECTAI_ALLOW_WEIGHT_DOWNLOAD=1 is set.

Exit codes: 0 success; 2 configuration error; 4 weights unavailable; 1 any other failure.
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Sequence

import torch

from inspectai.errors import ConfigError, InspectAIError, WeightsNotFoundError
from inspectai.models.features import (
    SUPPORTED_BACKBONES,
    fetch_weights,
    load_backbone,
    weights_cache_path,
)

EXIT_OK = 0
EXIT_FAILURE = 1
EXIT_CONFIG_ERROR = 2
EXIT_WEIGHTS_UNAVAILABLE = 4

logger = logging.getLogger("fetch_weights")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Fetch pretrained backbone weights.")
    parser.add_argument("--backbone", choices=tuple(SUPPORTED_BACKBONES), required=True)
    args = parser.parse_args(argv)
    try:
        path = weights_cache_path(args.backbone)
        if path.is_file():
            print(f"Already cached: {path}")
        else:
            print(f"Downloading {args.backbone} weights to {path} ...")
            try:
                fetch_weights(args.backbone)
            except (OSError, RuntimeError) as exc:
                raise WeightsNotFoundError(f"Download failed: {exc}") from exc
        _, info = load_backbone(args.backbone, allow_download=False)  # verifies the hash
        print(f"OK: {info.backbone} {info.identifier} sha256={info.sha256}")
        return EXIT_OK
    except ConfigError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return EXIT_CONFIG_ERROR
    except WeightsNotFoundError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return EXIT_WEIGHTS_UNAVAILABLE
    except InspectAIError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return EXIT_FAILURE


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    assert torch is not None
    try:
        sys.exit(main())
    except Exception:
        logger.exception("Unexpected failure")
        sys.exit(EXIT_FAILURE)
