from methods.evidence_qa import schema as qa_schema


NONE = '__no_evidence__'


def answer_schema(count):
    contract = qa_schema.answer(count)
    if count == 0:
        contract['properties']['evidence'] = {'type': 'array', 'maxItems': 0}
    return contract


