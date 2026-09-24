from contextlib import contextmanager

from torch.profiler import record_function

from jev_spawn.infra import finite_batch


@contextmanager
def instrument(backend, scopes):
    originals = {name: getattr(finite_batch, name) for name in scopes['functions']}
    active = []

    def wrapped(name, function):
        def call(*args, **kwargs):
            with record_function(scopes['functions'][name]):
                return function(*args, **kwargs)
        return call

    def enter(module, args, kwargs):
        scope = record_function(scopes['model'])
        active.append(scope)
        scope.__enter__()

    def leave(module, args, result):
        active.pop().__exit__(None, None, None)

    for name, function in originals.items():
        setattr(finite_batch, name, wrapped(name, function))
    handles = [backend.model.model.register_forward_pre_hook(enter, with_kwargs=True),
               backend.model.model.register_forward_hook(leave)]
    try:
        yield
    finally:
        for name, function in originals.items():
            setattr(finite_batch, name, function)
        for handle in handles:
            handle.remove()
