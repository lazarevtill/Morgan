"""Tests for the application state store."""

import threading

from morgan.app_state.store import AppState, AppStateStore


# --- AppState dataclass ---

class TestAppState:
    def test_defaults(self):
        s = AppState()
        assert s.verbose is False
        assert s.main_model == "qwen2.5:7b"
        assert s.permission_mode == "interactive"
        assert s.tasks == {}
        assert s.channels == {}
        assert s.plugins == {}
        assert s.skills == []
        assert s.status_text is None

    def test_custom(self):
        s = AppState(verbose=True, main_model="llama3", status_text="ready")
        assert s.verbose is True
        assert s.main_model == "llama3"
        assert s.status_text == "ready"


# --- AppStateStore ---

class TestAppStateStore:
    def test_get_returns_copy(self):
        store = AppStateStore()
        s1 = store.get_state()
        s1.verbose = True  # mutate the copy
        s2 = store.get_state()
        assert s2.verbose is False  # original unchanged

    def test_set_state(self):
        store = AppStateStore()
        store.set_state(lambda s: setattr(s, "verbose", True))
        assert store.get_state().verbose is True

    def test_subscribe_and_notify(self):
        store = AppStateStore()
        notifications: list[AppState] = []
        store.subscribe(lambda s: notifications.append(s))
        store.set_state(lambda s: setattr(s, "status_text", "booting"))
        assert len(notifications) == 1
        assert notifications[0].status_text == "booting"

    def test_unsubscribe(self):
        store = AppStateStore()
        notifications: list[AppState] = []
        unsub = store.subscribe(lambda s: notifications.append(s))
        store.set_state(lambda s: setattr(s, "verbose", True))
        unsub()
        store.set_state(lambda s: setattr(s, "verbose", False))
        assert len(notifications) == 1  # second change not received

    def test_multiple_subscribers(self):
        store = AppStateStore()
        a_calls: list[str] = []
        b_calls: list[str] = []
        store.subscribe(lambda s: a_calls.append(s.main_model))
        store.subscribe(lambda s: b_calls.append(s.main_model))
        store.set_state(lambda s: setattr(s, "main_model", "gpt4"))
        assert a_calls == ["gpt4"]
        assert b_calls == ["gpt4"]

    def test_initial_state(self):
        custom = AppState(verbose=True, main_model="custom")
        store = AppStateStore(initial=custom)
        assert store.get_state().verbose is True
        assert store.get_state().main_model == "custom"

    def test_thread_safety(self):
        store = AppStateStore()
        results: list[int] = []

        def bump(_s):
            _s.tasks["count"] = _s.tasks.get("count", 0) + 1

        def worker():
            for _ in range(100):
                store.set_state(bump)

        threads = [threading.Thread(target=worker) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert store.get_state().tasks["count"] == 400

    def test_unsubscribe_idempotent(self):
        store = AppStateStore()
        unsub = store.subscribe(lambda s: None)
        unsub()
        unsub()  # should not raise

    def test_set_state_complex_update(self):
        store = AppStateStore()

        def add_plugin(s):
            s.plugins["auth"] = {"enabled": True}
            s.skills.append("summarize")

        store.set_state(add_plugin)
        state = store.get_state()
        assert state.plugins == {"auth": {"enabled": True}}
        assert state.skills == ["summarize"]
