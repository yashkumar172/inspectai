"""Page 3: evaluation dashboard (SPEC.md Section 8).

Shows metrics only when valid evaluation results exist. Every number comes from the report files
written by `scripts/evaluate_model.py`; before a category is shown, its per-image predictions are
used to re-derive the image-level metrics, and the models must have been evaluated on identical
splits and settings. Invalid results are reported as such, with no metrics.
"""

from __future__ import annotations

import logging

import altair as alt
import pandas as pd
import streamlit as st

from inspectai.config import Paths
from inspectai.errors import InspectAIError
from inspectai.evaluation.dashboard import (
    FALSE_NEGATIVE,
    FALSE_POSITIVE,
    CategoryResults,
    confusion_matrix,
    metric_rows,
    misclassified_images,
    performance_rows,
)
from inspectai.inference.predictor import inspect_image, option_by_key
from inspectai.ui.common import (
    MSG_INFERENCE_FAILED,
    VISA_ATTRIBUTION,
    dashboard_results,
    environment,
    get_dataset,
    load_model,
)

logger = logging.getLogger("inspectai.app")

# Wide enough for the longest model name ("C PatchCore-inspired full (tiled)") at the chart's
# default font; the default axis label limit cut names off ("C PatchCore-insp...").
AXIS_LABEL_LIMIT_PX = 360

KIND_LABELS = {
    FALSE_POSITIVE: "False positives (normal images scored above the threshold)",
    FALSE_NEGATIVE: "False negatives (anomalous images scored at or below the threshold)",
}

NO_RESULTS = (
    "No evaluation results were found, so there are no metrics to show. Evaluate a model first, "
    "for example: `python scripts/evaluate_model.py --model gaussian --category cashew`, then "
    "reload this page."
)


def render() -> None:
    st.title("Evaluation dashboard")
    env = environment()
    if env.paths is None:
        st.error(env.message)
        return
    results = dashboard_results(env.paths)
    if not results:
        st.info(NO_RESULTS)
        return

    category = st.selectbox(
        "Category",
        list(results),
        format_func=lambda c: c if results[c].valid else f"{c} (results not valid)",
        key="dashboard_category",
    )
    _category(results[category], env.paths)
    st.caption(VISA_ATTRIBUTION)


def auroc_chart(auroc: dict[str, float]) -> alt.Chart:
    """Horizontal bars of image AUROC, with the full model names on the Y axis.

    The AUROC scale is fixed to 0..1 so that bar lengths can be compared across categories.
    """
    frame = pd.DataFrame({"Model": list(auroc), "Image AUROC": list(auroc.values())})
    return (
        alt.Chart(frame)
        .mark_bar()
        .encode(
            x=alt.X("Image AUROC:Q", scale=alt.Scale(domain=[0, 1])),
            y=alt.Y(
                "Model:N",
                sort=list(auroc),
                title=None,
                axis=alt.Axis(labelLimit=AXIS_LABEL_LIMIT_PX),
            ),
            tooltip=["Model", alt.Tooltip("Image AUROC:Q", format=".3f")],
        )
    )


def _category(result: CategoryResults, paths: Paths) -> None:
    if not result.valid:
        st.error(
            f"The results for {result.category} did not pass validation and are not shown. "
            f"{result.problem}"
        )
        st.caption("Re-run the evaluation (and the comparison) for this category to fix this.")
        return

    normal, anomalous = result.counts
    st.markdown(
        f"**Dataset split used for every number below:** {result.split_label}: "
        f"{normal} normal and {anomalous} anomalous images."
    )
    st.caption(
        "Thresholds come from held-out normal training images only. All models were evaluated on "
        "identical splits at the same evaluation resolution, and the image-level metrics were "
        "re-derived from the saved per-image predictions when this page loaded."
    )
    if result.not_measured:
        st.info("Not measured for this category: " + ", ".join(result.not_measured) + ".")

    st.subheader("Model comparison")
    st.dataframe(metric_rows(result), hide_index=True, width="stretch")
    st.caption(
        "Image AUROC and AP rank images from most to least anomalous (no threshold). Precision, "
        "recall, FPR and F1 are at each model's stored decision threshold. Pixel metrics are "
        "pooled over the test images at the evaluation resolution; IoU and Dice use a pixel "
        "threshold from normal calibration images. 'undefined' means the data cannot define the "
        "metric."
    )
    auroc = {
        r["label"]: r["metrics"]["image"]["auroc"]
        for r in result.records
        if r["metrics"]["image"]["auroc"] is not None
    }
    if auroc:
        st.altair_chart(auroc_chart(auroc), width="stretch")

    if result.paired:
        st.subheader("Paired AUROC differences")
        st.dataframe(
            [
                {
                    "A": d["a"],
                    "B": d["b"],
                    "AUROC(A) - AUROC(B)": (
                        "undefined"
                        if d["auroc_difference"] is None
                        else f"{d['auroc_difference']:+.3f}"
                    ),
                    "95% paired interval": (
                        "undefined"
                        if d["ci95"] is None
                        else f"{d['ci95'][0]:+.3f} to {d['ci95'][1]:+.3f}"
                    ),
                    "Interval excludes 0": (
                        "undefined"
                        if d["excludes_zero"] is None
                        else ("yes" if d["excludes_zero"] else "no")
                    ),
                }
                for d in result.paired
            ],
            hide_index=True,
            width="stretch",
        )
        st.caption(
            "All models scored the same test images, so differences are compared with a paired "
            "bootstrap. An interval that includes 0 means no difference is demonstrated."
        )

    st.subheader("Confusion matrix")
    labels = [r["label"] for r in result.records]
    chosen = st.selectbox("Model", labels, key="dashboard_confusion_model")
    record = result.records[labels.index(chosen)]
    matrix = confusion_matrix(record)
    st.table(
        [{"": truth, **counts} for truth, counts in matrix.items()],
    )
    st.caption(
        f"At the stored threshold of {record['metrics']['image']['threshold']:.3f} "
        f"({record['threshold']['image']['policy'].replace('_', ' ')})."
    )

    _error_analysis(result, paths)

    st.subheader("Inference performance")
    st.dataframe(performance_rows(result), hide_index=True, width="stretch")
    st.caption(
        "Measured on one machine during the evaluation runs; latency is the model's prediction "
        "time per image (image decoding excluded)."
        + (
            " Rows marked (tiled) are the same model fitted and scored with the faster "
            "`--index-backend tiled` kernels (same memory bank; see docs/benchmarks.md)."
            if result.tiled
            else ""
        )
    )

    if result.examples:
        st.subheader("Representative predictions")
        for example in result.examples:
            st.image(str(example.path), caption=example.caption, width="stretch")


def _error_analysis(result: CategoryResults, paths: Paths) -> None:
    """Triage the mistakes of one model: the scores always, the images when the dataset is here."""
    st.subheader("Error analysis")
    labels = [r["label"] for r in result.records]
    model_label = st.selectbox("Model", labels, key="dashboard_error_model")
    run = result.runs[labels.index(model_label)]
    errors = misclassified_images(run)
    st.caption(
        f"At the stored threshold, {model_label} made {len(errors[FALSE_POSITIVE])} false "
        f"positives and {len(errors[FALSE_NEGATIVE])} false negatives among "
        f"{len(run.predictions)} evaluated images. Read from the saved per-image predictions."
    )
    kind = st.radio(
        "Mistakes to inspect",
        list(KIND_LABELS),
        format_func=KIND_LABELS.__getitem__,
        key="dashboard_error_kind",
    )
    rows = errors[kind]
    if not rows:
        st.success("This model made no mistakes of this kind on these images.")
        return
    st.dataframe(
        [
            {
                "Image": r["name"],
                "Score": round(r["score"], 3),
                "Threshold": round(r["threshold"], 3),
                "Score / threshold": round(r["ratio"], 2),
                "Ground truth": "anomalous" if r["label"] else "normal",
            }
            for r in rows
        ],
        hide_index=True,
        width="stretch",
    )
    index = st.selectbox(
        "Look at one image",
        list(range(len(rows))),
        format_func=lambda i: f"{rows[i]['name']} (score {rows[i]['score']:.3f})",
        key="dashboard_error_sample",
    )
    _show_mistake(result.category, run.spec.report_folder, rows[index], paths)


def _show_mistake(category: str, option_key: str, row: dict, paths: Paths) -> None:
    truth = "anomalous" if row["label"] else "normal"
    decision = "Anomaly detected" if row["score"] > row["threshold"] else "Normal"
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Anomaly score", f"{row['score']:.3f}")
    c2.metric("Calibrated threshold", f"{row['threshold']:.3f}")
    c3.metric("Ground truth", truth)
    c4.metric("Model decision", decision)

    dataset = get_dataset(paths)
    if dataset is None:
        st.info(
            "The dataset images are not part of this deployment, so only the saved scores can "
            "be shown. Run the app with INSPECTAI_DATA_ROOT set to see the image and overlay."
        )
        return
    try:
        # only images the split file lists are ever opened, never a path taken from the report
        sample = next(s for s in dataset.get_samples(category) if s.image_rel == row["image"])
    except (StopIteration, InspectAIError):
        st.warning("This image is not listed in the local dataset.")
        return
    outcome = load_model(paths, option_by_key(option_key), category)
    if outcome.predictor is None:
        st.warning(outcome.message)
        return
    try:
        image = dataset.load_image(sample)
        mask = dataset.load_mask(sample, image.shape[:2]) if sample.is_anomalous else None
        inspection = inspect_image(outcome.predictor, image)
    except (InspectAIError, ValueError, RuntimeError):
        logger.exception("The error-analysis image could not be processed")
        st.error(MSG_INFERENCE_FAILED)
        return
    a, b, c = st.columns(3)
    a.image(image, caption="Original image", width="stretch")
    b.image(inspection.overlay, caption="Anomaly overlay", width="stretch")
    if mask is not None:
        c.image(mask * 255, caption="Ground-truth defect mask", width="stretch")
    else:
        c.info("Normal image: no defect mask.")
    if abs(inspection.score - row["score"]) > 1e-3 * max(1.0, abs(row["score"])):
        st.caption(
            f"The model reproduces a score of {inspection.score:.3f} today, which differs from "
            "the saved one; the saved score is the one evaluated above."
        )
    st.caption(
        "Overlay colors use the model's fixed scale (blue 0, pixel threshold in the middle, red "
        f"twice that). {VISA_ATTRIBUTION}"
    )
