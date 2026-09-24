import json
from pathlib import Path

from baselines.common.graph_finite_service import StableGraphFiniteService
from baselines.common.jevspawn_service import DecisionRequest


class TaskDecisionRequest(DecisionRequest):
    @property
    def signature(self):
        return 'finite'


class RecordedStableGraphFiniteService(StableGraphFiniteService):
    def __init__(self, backend, shared, deadlines, *, settings, prompts):
        self.recording = settings['finite_recording']
        self.recorded_cohorts = []
        super().__init__(backend, shared, deadlines, settings=settings, prompts=prompts)
        if backend.parallel_commands.is_leader:
            Path(self.recording['directory']).mkdir(parents=True, exist_ok=True)

    def _finite_score(self, batch, lengths):
        if self.backend.parallel_commands.is_leader:
            record = {'cohort': len(self.recorded_cohorts), 'requests': [
                {'task_id': request.task_id, 'field': request.field, 'messages': request.messages,
                 'rendered': request.admitted.rendered, 'tokens': list(request.admitted.tokens),
                 'input_tokens': len(request.admitted.tokens), 'base_length': length}
                for request, length in zip(batch, lengths, strict=True)]}
            path = Path(self.recording['directory'], self.recording['file'].format(index=record['cohort']))
            with path.open('x') as stream:
                json.dump(record, stream, **self.recording['serialization'])
                stream.write('\n')
            self.recorded_cohorts.append(path)
        return super()._finite_score(batch, lengths)


class RecordedTaskStableGraphFiniteService(RecordedStableGraphFiniteService):
    request_type = TaskDecisionRequest
