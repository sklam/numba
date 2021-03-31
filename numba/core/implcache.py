"""
Implementation cache used internally
"""
from numba.core import errors


class CachedPayloadError(errors.InternalError):
    pass


class UnexpectedPayloadError(CachedPayloadError):
    pass


class ImplCache:
    """A cache for saving compiled results.
    """

    dict_type = dict
    """
    The dictionary type can be overridden.
    """

    class Payload:
        def get_success(self):
            msg = "expecting successful compilation cache but missing"
            raise errors.UnexpectedPayloadError(msg)

        def get_failed(self):
            msg = "expecting failed compilation cache but missing"
            raise errors.UnexpectedPayloadError(msg)

    class Success(Payload):
        def __init__(self, data):
            self._data = data

        def get_success(self):
            return self._data

    class Failed(Payload):
        def __init__(self, data):
            self._data = data

        def get_failed(self):
            return self._data

    def __init__(self):
        self._data = self.dict_type()

    def add_success(self, key, impl):
        """Save a successful compilation
        """
        self._data[key] = self.Success(impl)

    def add_failed(self, key, failed):
        self._data[key] = self.Failed(failed)

    def get_expected(self, key):
        """
        Raises
        ------
        - `KeyError` is `key` is not defined.
        - `UnexpectedPayloadError` if the `key` is defined but containing
          a "failed" payload.
        """
        return self._data[key].get_success()

    def __len__(self):
        return len(self._data)

    def __iter__(self):
        return iter(self._data)
