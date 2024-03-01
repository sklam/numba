import sys
from enum import IntEnum
from typing import Any
from collections import defaultdict
from numba._dispatcher import initialize_dict_watcher_once

# https://github.com/python/cpython/blob/a81d9509ee60c405e08012aec5d97da5840b590b/Include/cpython/dictobject.h#L73-L85


class DictWatchEvent(IntEnum):
    ADDED = 0
    MODIFIED = 1
    DELETED = 2
    CLONED = 3
    CLEARED = 4
    DEALLOCATED = 5


class DictWatcher:
    def process_event(
        self, event: DictWatchEvent, dct: dict, key: Any, new_value: Any
    ) -> None:
        """Dispatch on the given event.

        Invoked by DictWatcherManager.callback
        """
        if event == DictWatchEvent.ADDED:
            self.event_added(dct, key, new_value)
        elif event == DictWatchEvent.MODIFIED:
            self.event_modified(dct, key, new_value)
        elif event == DictWatchEvent.DELETED:
            self.event_deleted(dct, key)
        elif event == DictWatchEvent.CLONED:
            self.event_cloned(dct, key)
        elif event == DictWatchEvent.CLEARED:
            self.event_cleared(dct)
        elif event == DictWatchEvent.DEALLOCATED:
            self.event_deallocated(dct)
        else:
            raise ValueError(f"unknown DictWatchEvent: {event}")

    def event_added(self, dct, key, value) -> None:
        """Triggered when dictionary is inserting a new key-value pair.

        Parameters
        ----------
        dct:
            Watched dict
        key:
            Key being inserted
        value:
            Value being inserted
        """
        pass

    def event_modified(self, dct, key, value) -> None:
        """Triggered when dictionary is replacing a key with a new value.

        Parameters
        ----------
        dct:
            Watched dict
        key:
            Key being modified
        value:
            Value being inserted
        """
        pass

    def event_deleted(self, dct, key) -> None:
        """Triggered when dictionary is deleting a key.

        Parameters
        ----------
        dct:
            Watched dict
        key:
            Key being deleted
        """
        pass

    def event_cloned(self, dct, src_dct) -> None:
        """Triggered when ``dct.update(src_dct)`` is called on an empty
        dictionary.

        Parameters
        ----------
        dct:
            Watched dict
        src_dct: dict
            Source dictionary
        """
        pass

    def event_cleared(self, dct) -> None:
        """Triggered when dictionary is being cleared; e.g. ``dct.clear()``.

        Parameters
        ----------
        dct:
            Watched dict
        """
        pass

    def event_deallocated(self, dct) -> None:
        """Triggered when dictionary is being deleted.

        Parameters
        ----------
        dct:
            Watched dict
        """
        pass


class ChangeWatcher(DictWatcher):
    class FakeObject:
        def __init__(self, name):
            self._name = name

        def __repr__(self):
            return f"<{self._name}>"

    DELETED = FakeObject("DELETED")

    def __init__(self, keys):
        self._keys = frozenset(keys)

    def on_change(self, key: Any, value: Any):
        raise NotImplementedError

    def event_added(self, dct, key, value) -> None:
        if key in self._keys:
            self.on_change(key, value)

    def event_modified(self, dct, key, value) -> None:
        if key in self._keys:
            self.on_change(key, value)

    def event_deleted(self, dct, key) -> None:
        if key in self._keys:
            self.on_change(key, self.DELETED)

    def event_cloned(self, dct, src_dct) -> None:
        for key in self._keys:
            if key in src_dct:
                self.on_change(key, src_dct[key])

    def event_cleared(self, dct) -> None:
        for key in self._keys:
            if key in dct:
                self.on_change(key, self.DELETED)

    def event_deallocated(self, dct) -> None:
        self.event_cleared(dct)


class DictWatcherManager:
    _singleton = None
    _watchers: dict[int, DictWatcher]

    def __new__(cls):
        if cls._singleton is None:
            obj = object.__new__(cls)
            cls._singleton = obj

        return cls._singleton

    def __init__(self):
        self._watchers = defaultdict(list)
        # Get hold of the function because sys module can be deleted before
        # this watcher manager is gone.
        self._is_finalizing = sys.is_finalizing

    def __init_subclass__(cls) -> None:
        raise TypeError("cannot subclass DictWatcherManager")

    def watch(self, d: dict, user_cb: DictWatcher):
        """Add a watcher on the dictionary."""
        k = id(d)
        initialize_dict_watcher_once(self.callback, d)
        self._watchers[k].append(user_cb.process_event)

    def callback(self, event, dct, key, new_value):
        """
        Invoked by the callback for PyDict_AddWatcher().
        """
        if self._is_finalizing():
            return
        event = DictWatchEvent(event)
        for user_cb in self._watchers[id(dct)]:
            user_cb(event, dct, key, new_value)
        if event == DictWatchEvent.DEALLOCATED:
            self._remove_watcher(id(dct))

    def _remove_watcher(self, key):
        """Remove watcher by dictionary id."""
        del self._watchers[key]
