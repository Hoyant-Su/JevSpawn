import argparse
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import json
from pathlib import Path
import time

from transformers import AutoTokenizer
import yaml

from baselines.foldagent.adapter import TokenizerInterface
from jev_spawn.infra.configuration import CORE
from jev_spawn.runtime.tokenization import SynchronizedTokenizer


def qualify(config):
    shared = yaml.safe_load(Path(config['shared_config']).read_text())
    tokenizer = AutoTokenizer.from_pretrained(shared['model']['path'],
        padding_side=CORE['backend']['padding_side'], local_files_only=CORE['backend']['local_files_only'])
    original = TokenizerInterface(deepcopy(tokenizer))
    started = time.perf_counter()
    resource = SynchronizedTokenizer(tokenizer)
    initialization = time.perf_counter() - started
    candidate = TokenizerInterface(resource)
    batches = json.loads(Path(config['source_batches']).read_text())
    messages = [message for batch in batches for message in batch['messages']]
    reference = [original.apply_chat_template(message, **config['chat_options']) for message in messages]
    with ThreadPoolExecutor(max_workers=shared['runtime']['root_batch_size']) as pool:
        actual = list(pool.map(lambda message: candidate.apply_chat_template(
            message, **config['chat_options']), messages))
    assert actual == reference
    texts = [original.apply_chat_template(message, **config['render_options']) for message in messages]
    operations = [(text, options) for text in texts for options in config['tokenize_options']]
    expected = [dict(tokenizer(text, **options)) for text, options in operations]
    scheduler_state = (deepcopy(tokenizer.backend_tokenizer.padding),
                       deepcopy(tokenizer.backend_tokenizer.truncation))
    with ThreadPoolExecutor(max_workers=shared['runtime']['root_batch_size']) as pool:
        encoded = list(pool.map(lambda item: dict(resource(item[0], **item[1])), operations))
        decoded = list(pool.map(lambda ids: resource.decode(ids, **config['decode_options']), reference))
    assert encoded == expected
    assert decoded == [tokenizer.decode(ids, **config['decode_options']) for ids in reference]
    assert scheduler_state == (tokenizer.backend_tokenizer.padding, tokenizer.backend_tokenizer.truncation)
    assert resource.tokenizer is not tokenizer
    result = {'source_batches': config['source_batches'], 'model': shared['model']['path'],
        'concurrent_workers': shared['runtime']['root_batch_size'], 'message_sequences': len(messages),
        'exact_chat_token_ids': True, 'mixed_setting_calls': len(operations), 'exact_mixed_setting_tokens': True,
        'exact_decoded_text': True, 'scheduler_tokenizer_state_unchanged': True,
        'resource_initialization_seconds': initialization,
        'scope': 'Complete saved message workload, concurrent real tokenization. No model inference or speed claim.'}
    Path(config['result']).write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    qualify(json.loads(parser.parse_args().config.read_text()))
