from baselines.common.jevspawn_service import DecisionRequest
from baselines.common.recorded_finite_service import RecordedStableGraphFiniteService


class ScoredDecisionRequest(DecisionRequest):
    pass


class ScoredFiniteService(RecordedStableGraphFiniteService):
    def decide_scores(self, fields, *, task_id):
        fields = self.prepare_task_fields(self.prepare_state_fields(fields))
        return self.enqueue_decisions(fields, task_id=task_id, request_type=ScoredDecisionRequest)

    def _generate(self, batch):
        values = super()._generate(batch)
        return [dict(decision=value, logits=self.finite_tail.last_logits[index, :len(request.field['options'])].clone())
                if isinstance(request, ScoredDecisionRequest) else value
                for index, (request, value) in enumerate(zip(batch, values, strict=True))]
