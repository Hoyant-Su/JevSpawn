class FlowInputError(ValueError):
    """A model-generated program or value violates its input contract."""


class TaskLimitError(RuntimeError):
    pass


class InputLimitError(ValueError):
    pass


class InvalidOutputError(ValueError):
    pass
