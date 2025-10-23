from audacity_session import AudacityLabelSession


def test_add_marker(tmp_path):
    audio = tmp_path / "clip.wav"
    audio.write_text("dummy")
    sess = AudacityLabelSession(str(audio))
    sess.add_marker(1.234, "test")
    content = sess.label_path.read_text()
    assert "1.234\t1.234\ttest" in content
    # markers persist after reloading
    sess2 = AudacityLabelSession(str(audio))
    assert sess2.markers == [(1.234, 1.234, "test")]
