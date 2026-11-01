"""The dashboard's AUROC bar chart: full model names, nothing clipped, honest scale."""

import json

from inspectai.ui.dashboard_page import AXIS_LABEL_LIMIT_PX, auroc_chart

LABELS = {
    "A classical": 0.70,
    "B Gaussian dev": 0.92,
    "B Gaussian full": 0.88,
    "C PatchCore-inspired dev": 0.955,
    "C PatchCore-inspired full": 0.968,
}


def _spec() -> dict:
    return json.loads(auroc_chart(LABELS).to_json())


def test_the_y_axis_label_limit_leaves_room_for_the_longest_model_name() -> None:
    axis = _spec()["encoding"]["y"]["axis"]
    assert axis["labelLimit"] == AXIS_LABEL_LIMIT_PX
    # about 7 px per character at the default font; "C PatchCore-inspired full (tiled)" is 33
    longest = max([*LABELS, "C PatchCore-inspired full (tiled)"], key=len)
    assert AXIS_LABEL_LIMIT_PX >= 7 * len(longest)


def test_every_full_model_name_is_in_the_data_in_the_given_order() -> None:
    spec = _spec()
    (rows,) = spec["datasets"].values()
    assert [r["Model"] for r in rows] == list(LABELS)
    assert [r["Image AUROC"] for r in rows] == list(LABELS.values())
    assert spec["encoding"]["y"]["sort"] == list(LABELS)  # the table's order, not alphabetical
    assert spec["encoding"]["y"]["field"] == "Model"


def test_the_auroc_axis_is_fixed_to_zero_to_one() -> None:
    assert _spec()["encoding"]["x"]["scale"]["domain"] == [0, 1]
