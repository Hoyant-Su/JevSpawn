from baselines.official.model_service import GenerationService


class FoldService(GenerationService):
    def _generate(self, batch):
        texts = super()._generate(batch)
        return [{'text': text, 'token_ids': token_ids}
                for text, token_ids in zip(texts, self.records[-1]['output_token_ids'])]
