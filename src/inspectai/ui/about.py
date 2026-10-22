"""Page 4: about and limitations (SPEC.md Section 8)."""

from __future__ import annotations

import streamlit as st

from inspectai.ui.common import VISA_ATTRIBUTION


def render() -> None:
    st.title("About and limitations")
    st.warning(
        "InspectAI is a research and educational prototype. It is not a certified industrial "
        "quality-control system and must not be used for safety-critical decisions."
    )

    st.markdown("### Anomaly detection versus supervised classification")
    st.write(
        "A supervised classifier needs labeled examples of every class it should recognize, "
        "including defective products, and it can only recognize defects that resemble its "
        "training examples. An anomaly detector is fitted on normal images only and measures how "
        "far a new image is from normal. That suits manufacturing, where defects are rare and "
        "varied, but it also means the detector can flag harmless variation (a different "
        "position, lighting, or an unseen but acceptable texture) as unusual."
    )

    st.markdown("### How an anomaly map is created")
    st.write(
        "Every model compares an image with what normal images looked like and produces a "
        "score for each region. The scores form the anomaly map; the image score is the maximum "
        "of the map; the decision is `score > threshold`. The threshold is chosen per model and "
        "per category from held-out normal training images (their 99th-percentile score), never "
        "from the test images."
    )
    st.markdown(
        """
| Model | How regions are compared with normal |
|---|---|
| **A: classical baseline** | The mean and spread of each pixel's color over the normal images; a region is unusual when its color is many standard deviations from the mean at the same position. Works only when the object is always at the same place. |
| **B: Gaussian model (PaDiM-inspired)** | Features from a frozen pretrained CNN (layer2 and layer3). At each position, a Gaussian over the normal images' features; the score is the Mahalanobis distance. |
| **C: PatchCore-inspired** | The same features, averaged over 3x3 neighbors, are stored as a memory bank of normal patches (reduced by greedy coreset selection to 10%); the score is the distance to the nearest stored patch. |
"""
    )

    st.markdown("### Pretrained weights and algorithms")
    st.write(
        "Models B and C use torchvision ImageNet-1k weights (`IMAGENET1K_V1`) without any "
        "fine-tuning: **ResNet-18** for the `dev` profile and **Wide ResNet-50-2** for the `full` "
        "profile, at 224x224 input. The weights are used frozen, only for feature extraction, and "
        "must already be on this computer (the app never downloads them). The algorithms are "
        "inspired by PaDiM (Defard et al., 2020) and PatchCore (Roth et al., CVPR 2022); they are "
        "simplified, CPU-friendly versions, not faithful reimplementations. Deviations are listed "
        "in the project README."
    )

    st.markdown("### Known failure cases")
    st.markdown(
        """
- **Small, faint, or low-contrast defects are often missed.** The decision thresholds are
  conservative (about 1% false alarms on normal images), so recall at the threshold is moderate
  even when the ranking of images is good.
- **The wrong category gives meaningless scores.** Scores are only meaningful if the object
  matches the selected category; an image of anything else is simply "far from normal".
- **Position, orientation, and lighting.** The classical baseline fails when the object moves;
  the deep models tolerate small shifts but can flag large changes in pose or illumination.
- **Image proportions.** Uploads whose aspect ratio differs from the category's by more than 25%
  are rejected, because the models were only fitted on images with those proportions.
- **Heatmaps are approximate.** They are low-resolution maps upsampled to the image size, so
  they indicate regions, not exact defect outlines.
- **Models disagree.** On some categories (for example `fryum`) the models separate defective
  from normal images much less well than on others; see the evaluation dashboard.
"""
    )

    st.markdown("### Dataset limitations")
    st.markdown(
        """
- The data are from **VisA**, an industrial-style benchmark, not footage from a production line.
  It includes food items as well as printed circuit boards.
- Each category has about 400 normal training images, and its test set has 50 normal and 100
  anomalous images, so metrics carry sampling uncertainty (the dashboard shows confidence
  intervals).
- One model is fitted per category; there is no single model for all objects.
- Only some categories have been fitted and evaluated (see the home page and the dashboard).
"""
    )
    st.caption(VISA_ATTRIBUTION)
    st.caption(
        'Citation: Zou, Jeong, Pemula, Zhang, and Dabeer, "SPot-the-Difference Self-Supervised '
        'Pre-training for Anomaly Detection and Segmentation", ECCV 2022 (arXiv:2207.14315).'
    )
