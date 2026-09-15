from qtpy.QtWidgets import QApplication

from ui.widgets import BatchProcessingWidget


_APP = QApplication.instance() or QApplication([])


def test_batch_widget_exposes_enabled_network_option():
    widget = BatchProcessingWidget()

    assert widget._network_checkbox.text() == "Calculate network metrics"
    assert widget._network_checkbox.isChecked()

    widget.close()


def test_batch_progress_tracks_statuses_separately():
    widget = BatchProcessingWidget()

    widget._on_progress((1, 3, "first.tif", "success"))
    widget._on_progress((2, 3, "second.tif", "partial"))
    widget._on_progress((3, 3, "third.tif", "failed"))

    assert widget._batch_success_count == 1
    assert widget._batch_partial_count == 1
    assert widget._batch_failed_count == 1
    assert widget.status_lbl.text() == "Processed 3/3: third.tif"

    widget.close()
