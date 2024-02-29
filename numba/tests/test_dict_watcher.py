from typing import Any

from numba.tests.support import TestCase
from numba.misc.dict_watcher import (
    DictWatcherManager,
    DictWatcher,
    ChangeWatcher,
)


class TestDictWatcher(TestCase):
    def test_dict_watcher(self):
        d = {}
        dct_id = id(d)
        watcher_manager = DictWatcherManager()
        events = []

        class TestDictWatcher(DictWatcher):
            def event_added(self, dct, key, value):
                events.append(("ADDED", dct, key, value))

            def event_modified(self, dct, key, value):
                events.append(("MODIFIED", dct, key, value))

            def event_deleted(self, dct, key):
                events.append(("DELETED", dct, key))

            def event_cloned(self, dct, src_dct):
                events.append(("CLONED", dct, src_dct))

            def event_cleared(self, dct):
                events.append(("CLEARED", dct))

            def event_deallocated(self, dct):
                # avoid retaining dct
                events.append(("DEALLOCATED", id(dct)))

        watcher = TestDictWatcher()
        watcher_manager.watch(d, watcher)

        d["a"] = 1
        self.assertEqual(events.pop(), ("ADDED", d, "a", 1))

        d["a"] = 2
        self.assertEqual(events.pop(), ("MODIFIED", d, "a", 2))

        # this won't make new event because the value is not modified
        d["a"] = 1 + 1
        self.assertFalse(events)

        del d["a"]
        self.assertEqual(events.pop(), ("DELETED", d, "a"))

        d.clear()
        self.assertEqual(events.pop(), ("CLEARED", d))

        src_dct = {"a": 1, "b": 2}
        d.update(src_dct)
        self.assertEqual(events.pop(), ("CLONED", d, src_dct))

        src_dct = {"c": 1, "d": 2}
        d.update(src_dct)
        self.assertEqual(events.pop(), ("ADDED", d, "d", 2))
        self.assertEqual(events.pop(), ("ADDED", d, "c", 1))

        del d
        self.assertEqual(events.pop(), ("DEALLOCATED", dct_id))

    def test_change_watcher(self):
        d = {}
        watcher_manager = DictWatcherManager()
        changes = []

        class TestChangeWatcher(ChangeWatcher):
            def on_change(self, key: Any, value: Any):
                changes.append((key, value))

        change_watcher = TestChangeWatcher(keys={'a'})
        watcher_manager.watch(d, change_watcher)

        d["a"] = 1
        self.assertEqual(changes.pop(), ("a", 1))
        d["a"] = 2
        self.assertEqual(changes.pop(), ("a", 2))

        # this won't make new event because the value is not modified
        d["a"] = 1 + 1
        self.assertFalse(changes)

        del d["a"]
        self.assertEqual(changes.pop(), ("a", change_watcher.DELETED))
        self.assertFalse(changes)

        d["a"] = 3
        self.assertEqual(changes.pop(), ("a", 3))

        d.clear()
        self.assertEqual(changes.pop(), ("a", change_watcher.DELETED))
        self.assertFalse(changes)

        src_dct = {"a": 1, "b": 2}
        d.update(src_dct)
        self.assertEqual(changes.pop(), ("a", 1))
        self.assertFalse(changes)

        src_dct = {"c": 1, "d": 2}
        d.update(src_dct)
        self.assertFalse(changes)

        d["a"] = 4
        self.assertEqual(changes.pop(), ("a", 4))

        del d
        self.assertEqual(changes.pop(), ("a", change_watcher.DELETED))
