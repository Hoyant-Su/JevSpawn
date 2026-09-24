import json


def state(row, document, arm, identity_mode):
    assert identity_mode in {"source", "ordinal"}
    identity = document["document_id"] if identity_mode == "source" else document["ordinal"]
    messages = [{'field_id': item['field_id'], 'choice': item['choice']}
                for item in document['results']]
    if arm == 'full_posterior':
        for message, result in zip(messages, document['results']):
            message.update(option_ids=result['option_ids'], probabilities=result['probabilities'])
    return json.dumps({'query': row['query'], 'document_id': identity,
                       'questions': row['fields'], 'worker_messages': messages},
                      ensure_ascii=False, separators=(',', ':'))


def groups(row, documents, arm, prompts, identity_mode):
    return [[{'id': document['document_id'], 'question': prompts['relevance_question'],
              'options': prompts['relevance_options'], 'state': state(row, document, arm, identity_mode)}]
            for document in documents]
