from morgan_brain.interfaces.events import Event, EventType


def test_event_has_schema_version_default():
    e = Event(type=EventType.MESSAGE_RECEIVED, user_id="u1")
    assert e.schema_version == 1


def test_event_schema_version_is_overridable():
    e = Event(type=EventType.MESSAGE_RECEIVED, user_id="u1", schema_version=2)
    assert e.schema_version == 2
