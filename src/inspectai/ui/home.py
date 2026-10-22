"""Page 1: home (SPEC.md Section 8)."""

from __future__ import annotations

import streamlit as st

from inspectai.data.dataset import SUPPORTED_OBJECTS
from inspectai.inference.predictor import available_categories, model_options
from inspectai.inference.uploads import ALLOWED_EXTENSIONS, MAX_UPLOAD_BYTES
from inspectai.ui.common import environment


def render() -> None:
    st.title("InspectAI")
    st.subheader("Industrial visual anomaly detection")
    st.write(
        "InspectAI finds defective products in photographs, shows where the defect is, and "
        "reports how it measured up against known-good products. It is a research and "
        "educational prototype built on the VisA benchmark; it runs on an ordinary CPU."
    )

    st.markdown("### What is anomaly detection?")
    st.write(
        "A normal image classifier learns from examples of both good and defective products. "
        "On a production line, defects are rare and come in unpredictable shapes, so there are "
        "often no defective examples to learn from. Anomaly detection learns only what "
        "**normal** products look like and flags anything that deviates. Each image gets an "
        "anomaly score and a heatmap that highlights the unusual regions."
    )

    st.markdown("### Fitted models and categories")
    env = environment()
    if env.paths is None:
        st.warning(env.message)
    else:
        rows = []
        for option in model_options():
            categories = available_categories(env.paths.artifact_dir, option)
            rows.append(
                {
                    "Model": option.label,
                    "Categories with a fitted model": ", ".join(categories) or "none",
                }
            )
        if any(r["Categories with a fitted model"] != "none" for r in rows):
            st.dataframe(rows, hide_index=True, width="stretch")
        else:
            st.warning(
                "No fitted models were found, so the inspection page cannot make predictions yet. "
                "See the README for how to fit one."
            )
    st.caption(
        "Supported categories (VisA objects): "
        + ", ".join(f"`{c}`" for c in SUPPORTED_OBJECTS)
        + "."
    )

    st.markdown("### How to inspect an image")
    types = ", ".join(e.lstrip(".").upper() for e in ALLOWED_EXTENSIONS)
    st.markdown(
        f"""
1. Open **Inspect an image** in the sidebar.
2. Choose a model and the **category** that matches the object in your photo.
3. Upload a {types} image (up to {MAX_UPLOAD_BYTES // 1024**2} MB), or pick a test image from the
   local dataset if it is available.
4. Press **Run inspection** and read the decision, the score next to its threshold, the heatmap,
   and the overlay. You can download the visualization and the prediction details.
"""
    )

    st.markdown("### Limitations and intended use")
    st.warning(
        "Scores are only meaningful if the uploaded object matches the selected category. "
        "This is **not** a certified industrial quality-control system and must not be used for "
        "safety-critical decisions."
    )
    st.markdown(
        """
- The anomaly score is a distance-like number, **not a probability**. Its scale differs between
  models and categories; compare it with that model's own threshold and normal range.
- Thresholds are conservative: few false alarms, but many subtle defects are missed.
- Models were fitted per category on one benchmark; other objects, cameras, or lighting are
  outside what was evaluated. See **About and limitations** for details.
"""
    )
