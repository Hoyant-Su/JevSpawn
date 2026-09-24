import json
from threading import Lock

import jsonschema

from baselines.common.errors import TaskLimitError
from baselines.tool_agents.tools import ActionError
from environments.arguments import typed_value


class TransitionBudget:
    """Count observed state transitions along each actual environment branch."""

    __slots__ = ('_environment', 'max_turns', 'submission_tool', 'transition_depth', 'transitions', '_lock')

    def __init__(self, environment, max_turns, submission_tool, transition_depth, transitions):
        self._environment = environment
        self.max_turns = max_turns
        self.submission_tool = submission_tool
        self.transition_depth = transition_depth
        self.transitions = transitions
        self._lock = Lock()

    def __getattr__(self, name):
        return getattr(self._environment, name)

    def __setattr__(self, name, value):
        if name in self.__slots__:
            object.__setattr__(self, name, value)
        else:
            setattr(self._environment, name, value)

    def fork(self):
        with self._lock:
            return type(self)(self._environment.fork(), self.max_turns, self.submission_tool,
                              self.transition_depth, self.transitions)

    def _advance(self, name):
        self.transition_depth += 1
        self.transitions.append({'depth': self.transition_depth, 'tool': name})

    def _reject(self, name, arguments, error):
        if self.transition_depth >= self.max_turns:
            raise TaskLimitError('Shared state-transition round budget exhausted; submission remains available.')
        message = error.message if isinstance(error, jsonschema.ValidationError) else str(error)
        result = {'error': {'type': type(error).__name__, 'message': message}, 'done': False}
        self._environment.actions.append({'tool': name, 'arguments': arguments, 'result': result})
        self._advance(name)
        return result

    def reject(self, name, arguments, error):
        """Record a malformed model command before native dispatch is possible."""
        with self._lock:
            return self._reject(name, arguments, error)

    def _invoke(self, operation, name, arguments, serialize_error):
        with self._lock:
            transition = name != self.submission_tool
            if transition and self.transition_depth >= self.max_turns:
                raise TaskLimitError('Shared state-transition round budget exhausted; submission remains available.')
            try:
                if name not in self._environment.tools:
                    raise ActionError(f'Unavailable tool: {name}')
                schema = self._environment.input_schemas[name]
                try:
                    normalized = typed_value(arguments, schema)
                except json.JSONDecodeError as error:
                    raise ActionError(str(error)) from error
                jsonschema.Draft202012Validator(schema).validate(normalized)
                result = operation(name, normalized)
            except (ActionError, jsonschema.ValidationError) as error:
                result = self._reject(name, arguments, error)
                return (json.dumps(result), False) if serialize_error else result
            if transition:
                self._advance(name)
            return result

    def execute(self, name, arguments):
        return self._invoke(self._environment.execute, name, arguments, True)

    def observe(self, name, arguments):
        return self._invoke(self._environment.observe, name, arguments, False)
