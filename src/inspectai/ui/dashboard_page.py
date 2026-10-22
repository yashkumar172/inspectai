"""Page 3: evaluation dashboard (SPEC.md Section 8).

Shows metrics only when valid evaluation results exist. Every number comes from the report files
written by `scripts/evaluate_model.py`; before a category is shown, its per-image predictions are
used to re-derive the image-level metrics, and the models must have been evaluated on identical
splits and settings. Invalid results are reported as such, with no metrics.
"""

from __future__ import annotations

import streamlit as st

from inspectai.evaluation.dashboard import (
    CategoryResults,
    confusion_matrix,
    metric_rows,
    performance_rows,
)
from inspectai.ui.common import VISA_ATTRIBUTION, dashboard_results, environment

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
    _category(results[category])
    st.caption(VISA_ATTRIBUTION)


def _category(result: CategoryResults) -> None:
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
        st.bar_chart(auroc, y_label="Image AUROC", x_label="Model", horizontal=True)

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

    st.subheader("Inference performance")
    st.dataframe(performance_rows(result), hide_index=True, width="stretch")
    st.caption(
        "Measured on one machine during the evaluation runs; latency is the model's prediction "
        "time per image (image decoding excluded)."
    )

    if result.examples:
        st.subheader("Representative predictions")
        for example in result.examples:
            st.image(str(example.path), caption=example.caption, width="stretch")
