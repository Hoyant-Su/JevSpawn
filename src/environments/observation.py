import re
from string import Formatter


class FormattedObservation:
    """Recover public fields from the environment's own display template."""

    def __init__(self, template):
        self.template = template
        parts = list(Formatter().parse(template))
        assert all(not spec and conversion is None for _, _, spec, conversion in parts)
        self.pattern = re.compile(''.join(
            re.escape(literal) + (f'(?P<{field}>.*?)' if field is not None else '')
            for literal, field, _, _ in parts), re.DOTALL)

    def fields(self, observation):
        match = self.pattern.fullmatch(observation)
        assert match is not None, 'Native observation differs from its declared display template.'
        values = match.groupdict()
        assert self.template.format(**values) == observation
        return values
