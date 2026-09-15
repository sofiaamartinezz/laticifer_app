import pytest

from utils.quantification import uses_tissue_reference


@pytest.mark.parametrize(
    "selection,expected",
    [
        ("Tissue area (auto)", True),
        ("Whole image", False),
    ],
)
def test_density_reference_selection_matches_visible_label(selection, expected):
    assert uses_tissue_reference(selection) is expected
