"""Page 2: inspect an image (SPEC.md Section 8).

The user uploads an image (or picks a test image from the local dataset), chooses a model and a
category, and runs the model. A prediction is only ever shown if a fitted model was loaded
through the artifact store and actually ran on the image; otherwise the page says
`Model unavailable` or explains the failure, and nothing is made up.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import streamlit as st
from numpy.typing import NDArray

from inspectai.data.dataset import SUPPORTED_OBJECTS, Sample
from inspectai.errors import InspectAIError, UploadError
from inspectai.inference.predictor import (
    NOT_A_PROBABILITY,
    STATUS_ANOMALY,
    STATUS_UNAVAILABLE,
    WRONG_CATEGORY_WARNING,
    Inspection,
    ModelOption,
    Predictor,
    available_categories,
    details_json,
    inspect_image,
    model_options,
    panel_png,
)
from inspectai.inference.uploads import ALLOWED_EXTENSIONS, MAX_UPLOAD_BYTES, validate_upload
from inspectai.preprocessing.images import ImageArray
from inspectai.ui.common import (
    MSG_INFERENCE_FAILED,
    VISA_ATTRIBUTION,
    LoadOutcome,
    clear_model_cache,
    dataset_test_images,
    environment,
    fit_hint,
    get_dataset,
    load_model,
)

logger = logging.getLogger("inspectai.app")

SOURCE_UPLOAD = "Upload an image"
SOURCE_DATASET = "Choose a test image from the local dataset"
DEFAULT_OPTION_KEY = "patchcore"  # highest mean image AUROC of the `dev` models in the comparison
STATE_KEY = "inspectai_shown"
FAILURE_KEY = "inspectai_load_failure"


@dataclass
class Shown:
    """What the page displays after a successful run (kept across reruns, e.g. for downloads)."""

    predictor: Predictor
    inspection: Inspection
    image: ImageArray
    source: str
    from_dataset: bool
    ground_truth: str | None
    mask: NDArray[np.uint8] | None


def _option_label(option: ModelOption, available: dict[str, list[str]]) -> str:
    return option.label if available[option.key] else f"{option.label} (not fitted)"


def render() -> None:
    st.title("Inspect an image")
    env = environment()
    if env.paths is None:
        st.error(f"**{STATUS_UNAVAILABLE}.** {env.message}")
        return
    paths = env.paths
    options = model_options()
    available = {o.key: available_categories(paths.artifact_dir, o) for o in options}
    if not any(available.values()):
        st.error(
            f"**{STATUS_UNAVAILABLE}.** No fitted models were found, so no prediction can be made."
        )
        st.info(
            "Fit a model first, for example: `python scripts/fit_model.py --model classical --category cashew`"
        )
        return

    keys = [o.key for o in options]
    by_key = {o.key: o for o in options}
    default_key = (
        DEFAULT_OPTION_KEY
        if available.get(DEFAULT_OPTION_KEY)
        else next(k for k in keys if available[k])
    )
    left, right = st.columns(2)
    with left:
        option_key = st.selectbox(
            "Model",
            keys,
            index=keys.index(default_key),
            format_func=lambda k: _option_label(by_key[k], available),
            key="inspect_model",
        )
    option = by_key[option_key]
    with right:
        category = st.selectbox(
            "Category (the kind of object in the image)",
            list(SUPPORTED_OBJECTS),
            index=(
                SUPPORTED_OBJECTS.index(available[option_key][0]) if available[option_key] else 0
            ),
            format_func=lambda c: c if c in available[option_key] else f"{c} (not fitted)",
            key="inspect_category",
        )
    st.warning(WRONG_CATEGORY_WARNING)

    if category not in available[option_key]:
        st.error(
            f"**{STATUS_UNAVAILABLE}.** No fitted '{option.label}' model exists for {category}."
        )
        st.info(f"Create it with: `{fit_hint(option, category)}`")
        return

    dataset = get_dataset(paths)
    sources = [SOURCE_UPLOAD] + ([SOURCE_DATASET] if dataset is not None else [])
    source = st.radio("Image source", sources, horizontal=True, key="inspect_source")

    uploaded = None
    chosen: Sample | None = None
    if source == SOURCE_UPLOAD:
        uploaded = st.file_uploader(
            f"Image ({', '.join(e.lstrip('.').upper() for e in ALLOWED_EXTENSIONS)}; maximum "
            f"{MAX_UPLOAD_BYTES // 1024**2} MB)",
            type=[e.lstrip(".") for e in ALLOWED_EXTENSIONS],
            max_upload_size=MAX_UPLOAD_BYTES // 1024**2,
            key="inspect_upload",
        )
    else:
        assert dataset is not None
        samples = dataset_test_images(dataset, category)
        if not samples:
            st.info("The local dataset has no test images for this category.")
        else:
            chosen = st.selectbox(
                "Test image",
                samples,
                format_func=lambda s: s.image_rel.split("/")[-1],
                key="inspect_sample",
            )

    can_run = uploaded is not None if source == SOURCE_UPLOAD else chosen is not None
    if st.button("Run inspection", type="primary", disabled=not can_run, key="inspect_run"):
        _run(paths, option, category, source, uploaded, chosen, dataset)

    failure = st.session_state.get(FAILURE_KEY)
    if failure is not None:
        st.error(f"**{STATUS_UNAVAILABLE}.** {failure}")
        if st.button("Try loading again", key="inspect_retry"):
            clear_model_cache()
            st.session_state.pop(FAILURE_KEY, None)
            st.rerun()
    shown = st.session_state.get(STATE_KEY)
    if shown is not None:
        _show(shown)


def _run(paths, option, category, source, uploaded, chosen, dataset) -> None:  # type: ignore[no-untyped-def]
    st.session_state.pop(STATE_KEY, None)
    st.session_state.pop(FAILURE_KEY, None)
    with st.spinner("Loading the model and checking its integrity..."):
        outcome: LoadOutcome = load_model(paths, option, category)
    if outcome.predictor is None:
        st.session_state[FAILURE_KEY] = outcome.message  # shown by render(), with a retry button
        return
    predictor = outcome.predictor

    ground_truth: str | None = None
    mask = None
    if source == SOURCE_UPLOAD:
        assert uploaded is not None
        try:
            image = validate_upload(
                uploaded.getvalue(), uploaded.name, predictor.reference_aspect_ratio
            )
        except UploadError as exc:
            st.error(f"This image cannot be used. {exc}")
            return
        origin = "upload"
    else:
        assert chosen is not None and dataset is not None
        try:
            image = dataset.load_image(chosen)
            mask = dataset.load_mask(chosen, image.shape[:2]) if chosen.is_anomalous else None
        except InspectAIError:
            logger.exception("A dataset image could not be loaded")
            st.error("The selected dataset image could not be loaded.")
            return
        origin = chosen.image_rel
        ground_truth = "anomalous" if chosen.is_anomalous else "normal"

    with st.spinner("Running the model..."):
        try:
            inspection = inspect_image(predictor, image)
        except (InspectAIError, ValueError, RuntimeError):
            logger.exception("Inference failed")
            st.error(MSG_INFERENCE_FAILED)
            return
    st.session_state[STATE_KEY] = Shown(
        predictor, inspection, image, origin, source == SOURCE_DATASET, ground_truth, mask
    )


def _show(shown: Shown) -> None:
    result, predictor = shown.inspection, shown.predictor
    st.divider()
    st.subheader("Result")
    st.caption(f"Model: {predictor.option.label} · Category: {predictor.category}")
    if result.is_anomaly:
        st.error(f"**{STATUS_ANOMALY}**")
    else:
        st.success("**Normal**")

    cal = result.calibration
    c1, c2, c3 = st.columns(3)
    c1.metric("Anomaly score (not a probability)", f"{result.score:.3f}")
    c2.metric("Decision threshold", f"{result.threshold:.3f}")
    c3.metric(
        f"Normal images' scores (n = {cal.n}): min / median / max",
        f"{cal.minimum:.2f} / {cal.median:.2f} / {cal.maximum:.2f}",
    )
    st.write(result.score_vs_normal)
    st.caption(
        f"The threshold is the {predictor.threshold_policy.replace('_', ' ')} threshold stored with "
        f"this model for {predictor.category}; the image is flagged when its score is above it."
    )
    st.warning(WRONG_CATEGORY_WARNING)
    st.caption(NOT_A_PROBABILITY)

    a, b, c = st.columns(3)
    a.image(shown.image, caption="Original image", width="stretch")
    b.image(result.heatmap, caption="Anomaly heatmap", width="stretch")
    c.image(result.overlay, caption="Heatmap overlay", width="stretch")
    st.caption(
        "Heatmap colors use a fixed scale: blue is 0, the middle is the model's pixel threshold "
        "(the 99th percentile of normal pixels), and red is twice that. Red areas are where the "
        "model sees the most unusual appearance."
    )
    if shown.from_dataset:
        st.caption(f"Ground truth from the dataset: **{shown.ground_truth}**. {VISA_ATTRIBUTION}")
        if shown.mask is not None:
            st.image(shown.mask * 255, caption="Ground-truth defect mask", width="stretch")

    st.download_button(
        "Download visualization (PNG)",
        data=panel_png(shown.image, result),
        file_name=f"inspectai_{predictor.category}_{predictor.option.key}.png",
        mime="image/png",
        key="inspect_download_png",
    )
    st.download_button(
        "Download prediction details (JSON)",
        data=details_json(predictor, result, shown.image.shape[:2], shown.source),
        file_name=f"inspectai_{predictor.category}_{predictor.option.key}.json",
        mime="application/json",
        key="inspect_download_json",
    )
    st.caption(f"Inference took {result.seconds * 1000:.0f} ms.")
