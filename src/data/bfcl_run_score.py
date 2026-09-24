import jsonschema

from data.evaluate_bfcl import BFCLScorer


class BFCLRunScorer:
    def __init__(self, settings):
        self.settings = settings
        self.scorer = BFCLScorer(settings['evaluation'])

    def score(self, task, result, reference):
        assert task['task_id'] == result['task_id'] == reference['task_id']
        if result['status'] != self.settings['completed_status']:
            assert result['status'] in self.settings['noncompleted_statuses']
            return {'task_id': task['task_id'], 'score': self.settings['failure_score'],
                    'status': result['status']}
        jsonschema.validate(result['answer'], task['answer_schema'])
        native = self.scorer.score(task, result['answer']['completion'], reference)
        return {'task_id': task['task_id'], 'score': self.settings['validity_scores'][str(native['valid'])],
                'native': native}
