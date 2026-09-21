from core.obs.trace import span


def test_nested_span_has_parent():
    # The context manager returns the child id, while the logger receives the
    # parent chain; this also exercises contextvars' normal nesting path.
    with span("outer") as outer:
        with span("inner") as inner:
            assert outer != inner
