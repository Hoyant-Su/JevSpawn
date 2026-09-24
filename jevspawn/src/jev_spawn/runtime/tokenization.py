from copy import deepcopy
from functools import partial
from threading import RLock


class SynchronizedTokenizer:
    """Isolate mutable tokenizer settings from the model service and task threads."""

    def __init__(self, tokenizer):
        self.tokenizer = deepcopy(tokenizer)
        self.lock = RLock()

    def _call(self, name, *args, **kwargs):
        with self.lock:
            return getattr(self.tokenizer, name)(*args, **kwargs)

    def __call__(self, *args, **kwargs):
        return self._call('__call__', *args, **kwargs)

    def __getattr__(self, name):
        value = getattr(self.tokenizer, name)
        return partial(self._call, name) if callable(value) else value
