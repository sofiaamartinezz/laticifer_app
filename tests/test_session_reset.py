from types import MethodType, SimpleNamespace

from ui.transect_controller import (
    POINTS_LAYER_NAME,
    TRANSECT_LAYER_NAME,
    TransectController,
)
from ui.widgets import InteractiveEditorWidget
from data.settings import AppSettings


class FakeLayers(list):
    def __contains__(self, item):
        if isinstance(item, str):
            return any(layer.name == item for layer in self)
        return super().__contains__(item)

    def __getitem__(self, item):
        if isinstance(item, str):
            return next(layer for layer in self if layer.name == item)
        return super().__getitem__(item)

    def remove(self, item):
        if isinstance(item, str):
            item = self[item]
        super().remove(item)


class Recorder:
    def __init__(self):
        self.calls = 0

    def __call__(self, *args, **kwargs):
        self.calls += 1


def test_removing_source_image_resets_complete_session():
    image = SimpleNamespace(name="Source image")
    mask = SimpleNamespace(name="Laticifer mask")
    derived = SimpleNamespace(name="Diameter map")
    layers = FakeLayers([mask, derived])  # napari has already removed the image

    editor = SimpleNamespace(
        _resetting_session=False,
        _session_generation=4,
        settings=AppSettings(),
        base_image_layer=image,
        labels_layer=mask,
        viewer=SimpleNamespace(layers=layers),
        _transect_ctrl=SimpleNamespace(reset=Recorder()),
        dataset_root="dataset",
        initialized_from_model=True,
        um_per_px=0.5,
        scale_source="reference_line",
        scale_reference_pixels=100.0,
        scale_reference_length_um=50.0,
        last_transect_num_lines=25,
        last_transect_direction="both",
        last_transect_mean=3.0,
        tab_prepare=SimpleNamespace(clear_image=Recorder()),
        tab_density=SimpleNamespace(reset_results=Recorder()),
        tab_network=SimpleNamespace(reset_results=Recorder()),
        tab_mask=SimpleNamespace(set_ai_status=Recorder()),
        _tabs=SimpleNamespace(setCurrentIndex=Recorder()),
        _update_all_states=Recorder(),
    )

    editor._reset_session = MethodType(InteractiveEditorWidget._reset_session, editor)
    InteractiveEditorWidget._on_layer_removed(editor, SimpleNamespace(value=image))

    assert layers == []
    assert editor.base_image_layer is None
    assert editor.labels_layer is None
    assert editor.dataset_root is None
    assert editor.um_per_px is None
    assert editor.scale_source == "pixels_only"
    assert editor.scale_reference_pixels is None
    assert editor.scale_reference_length_um is None
    assert editor.last_transect_mean is None
    assert editor._session_generation == 5
    assert editor._transect_ctrl.reset.calls == 1
    assert editor.tab_prepare.clear_image.calls == 1
    assert editor.tab_density.reset_results.calls == 1
    assert editor.tab_network.reset_results.calls == 1
    assert editor._tabs.setCurrentIndex.calls == 1
    assert editor._update_all_states.calls == 1


def test_transect_reset_removes_layers_and_cached_state():
    layers = FakeLayers([
        SimpleNamespace(name=TRANSECT_LAYER_NAME),
        SimpleNamespace(name=POINTS_LAYER_NAME),
    ])
    state_changed = Recorder()
    controller = TransectController(
        SimpleNamespace(layers=layers), on_state_change=state_changed
    )
    controller._shapes_layer = object()
    controller.last_stats = {"mean": 3.0}
    controller._pending = True

    controller.reset()

    assert layers == []
    assert controller._shapes_layer is None
    assert controller.last_stats is None
    assert controller._pending is False
    assert state_changed.calls == 1
