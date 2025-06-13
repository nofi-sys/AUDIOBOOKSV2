import os
import threading
import time
from unittest import mock

import pytest
from gui import App
import ai_review

if not os.environ.get("DISPLAY"):
    pytest.skip("no display", allow_module_level=True)


def test_stop_button_interrupts_review():
    app = App()
    start_event = threading.Event()
    finish_event = threading.Event()

    def slow_review(path):
        ai_review._stop_review = False
        start_event.set()
        while not ai_review._stop_review:
            time.sleep(0.01)
        finish_event.set()
        return 0, 0

    with mock.patch("ai_review.review_file", side_effect=slow_review):
        t = threading.Thread(target=app._ai_review_worker)
        t.start()
        assert start_event.wait(1)
        app.stop_ai_review()
        t.join(timeout=1)
        assert finish_event.is_set()
    app.destroy()
