from typing import Any, Mapping, Iterator
from dataclasses import dataclass


@dataclass
class _Entry:
    key: Any
    value: Any
    attributes: Mapping

    def match(self, attrs: Mapping) -> bool:
        for k in attrs:
            if k in self.attributes:
                if self.attributes[k] == attrs[k]:
                    return True
        return False


class RegDB:
    _singleton = None
    _entries: list[_Entry]

    def __new__(cls):
        if cls._singleton is None:
            instance = cls._singleton = object.__new__(cls)
            instance._entries = []
        return cls._singleton

    def __len__(self) -> int:
        return len(self._entries)

    def insert(self, key, value, attributes) -> None:
        self._entries.append(_Entry(key, value, attributes))

    def query(self, key, attributes) -> Iterator[_Entry]:
        for ent in self._entries:
            if key == ent.key:
                if ent.match(attributes):
                    yield ent

    def filter_by_attrs(
        self, attributes, *, startpos: int = 0
    ) -> Iterator[tuple[int, _Entry]]:
        for i in range(startpos, len(self._entries)):
            ent = self._entries[i]
            if ent.match(attributes):
                yield i, ent


class DBListView:
    _attrs: Mapping
    _db: RegDB
    _view: list[_Entry]
    _size: int

    def __init__(self, attributes: Mapping):
        self._attrs = attributes
        self._db = RegDB()
        # localized view
        self._view = []
        self._size = 0
        self._lastpos = 0

    def _synchronize(self) -> int:
        iterator = self._db.filter_by_attrs(self._attrs, startpos=self._lastpos)
        for pos, ent in iterator:
            self._view.append(ent)
            self._size += 1
        self._lastpos = pos
        return self.checkpoint()

    def checkpoint(self) -> int:
        return self._size

    def get_updates(self, last_checkpoint) -> tuple[int, Iterator[_Entry]]:
        chkpt = self._synchronize()

        def iterator():
            for i in range(last_checkpoint, self._size):
                yield self._view[i]

        return chkpt, iterator()

    def append(self, key, value) -> None:
        self._db.insert(key, value, self._attrs)

    def __iter__(self) -> Iterator[_Entry]:
        self._synchronize()
        return iter(self._view)

    def __len__(self) -> int:
        self._synchronize()
        return self._size


def get_list_view(**attributes) -> DBListView:
    return DBListView(attributes)
