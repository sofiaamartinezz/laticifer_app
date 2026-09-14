from data.errors import user_error_message


def test_worker_exception_tuple_is_unwrapped():
    error = RuntimeError("model failed")

    assert user_error_message((RuntimeError, error, None)) == "model failed"


def test_permission_error_has_actionable_message():
    message = user_error_message(PermissionError("denied"))

    assert "writable" in message
