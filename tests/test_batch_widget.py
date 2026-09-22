from qtpy.QtWidgets import QApplication

from ui.widgets import BatchProcessingWidget, _batch_display_path


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


def test_batch_paths_hide_private_parent_folders_but_keep_real_paths():
    widget = BatchProcessingWidget()
    private_path = (
        "C:/Users/Researcher/Documents/Conference/laticifer_app/dataset/images"
    )

    widget.input_dir_edit.setText(private_path)

    assert widget.input_dir_edit.text() == "…/laticifer_app/dataset/images"
    assert widget.input_dir_edit.path() == private_path
    assert "Researcher" not in widget.input_dir_edit.text()
    assert _batch_display_path("C:/Users/Researcher/other/results") == "…/other/results"

    widget.close()


def test_batch_path_can_still_be_entered_manually():
    widget = BatchProcessingWidget()
    edited_path = "D:/work/laticifer_app/dataset/images"

    widget.input_dir_edit.set_path("C:/old/laticifer_app/dataset/images")
    widget.input_dir_edit.textEdited.emit(edited_path)
    widget.input_dir_edit.editingFinished.emit()

    assert widget.input_dir_edit.path() == edited_path
    assert widget.input_dir_edit.text() == "…/laticifer_app/dataset/images"

    widget.close()


def test_clearing_batch_path_does_not_retain_the_previous_real_path():
    widget = BatchProcessingWidget()
    widget.input_dir_edit.set_path("C:/private/laticifer_app/dataset/images")

    widget.input_dir_edit.clear()

    assert widget.input_dir_edit.text() == ""
    assert widget.input_dir_edit.path() == ""

    widget.close()
