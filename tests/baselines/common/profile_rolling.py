import argparse
import json
from pathlib import Path
import time

import torch

from baselines.common.config import SharedConfig
from baselines.common.tasks import read
from jev_spawn.infra.backend import Backend
from jev_spawn.runtime.decoding import CapturedDecode
from jev_spawn.runtime.rolling_decode import RollingDecode


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--shared-config', type=Path, required=True)
    parser.add_argument('--profile', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    config, profile = SharedConfig.load(args.shared_config), read(args.profile)
    args.output.mkdir(parents=True, exist_ok=False)
    backend = Backend(config.backend())
    requests = read(profile['requests'])[profile['batch_index']]['messages']
    assert len(requests) == config.runtime.batch_size
    size, steps, refill_step = len(requests), profile['tokens'], profile['refill_step']
    replaced = profile['replaced_rows']
    incoming = requests[:replaced]
    options = dict(do_sample=False, max_new_tokens=steps, pad_token_id=backend.tokenizer.pad_token_id)
    decoders, references = [], []
    with torch.inference_mode():
        for messages in (requests, incoming):
            texts = backend.tokenizer.apply_chat_template(messages, tokenize=False,
                                                          add_generation_prompt=True, enable_thinking=False)
            inputs = backend.tokenizer(texts, return_tensors='pt', padding=True,
                                       add_special_tokens=False).to(backend.device)
            required = inputs.input_ids.shape[1] + steps
            block = config.runtime.graph_cache_block_tokens
            decoder = CapturedDecode(backend, len(messages), (required + block - 1) // block * block)
            output, _, _ = decoder.generate_tokens(inputs, options,
                lambda ids, scores: torch.zeros(len(messages), device=backend.device, dtype=torch.bool),
                config.runtime.graph_warmup_steps)
            references.append(output[:, inputs.input_ids.shape[1]:].clone())
            decoder.prefill(inputs)
            decoders.append(decoder)
        rolling = RollingDecode(backend, size, max(d.capacity for d in decoders), decoders[0])
        rolling.load([(decoders[0], torch.arange(size, device=backend.device))])
        rolling.capture(config.runtime.graph_warmup_steps)
        emitted = []
        started = time.perf_counter()
        for step in range(steps):
            if step == refill_step:
                rolling.load([(rolling, torch.arange(replaced, size, device=backend.device)),
                              (decoders[1], torch.arange(replaced, device=backend.device))])
            emitted.append(rolling.ids[:, 0].clone())
            rolling.graph.replay()
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        actual = torch.stack(emitted, dim=1)
        expected = torch.cat([
            references[0][:, :refill_step],
            torch.cat([references[0][replaced:, refill_step:],
                       references[1][:, :steps - refill_step]])], dim=1)
        matches = (actual == expected).tolist()
        result = {'profile': profile, 'model': config.model.path, 'exact_token_parity': bool((actual == expected).all()),
                  'matching_tokens': sum(sum(row) for row in matches), 'total_tokens': actual.numel(),
                  'actual': actual.tolist(), 'expected': expected.tolist(), 'elapsed_seconds': elapsed,
                  'cache_transfer': 'GPU to GPU; complete KV, convolution and recurrent states',
                  'scope': 'Explicit fixed-length operator qualification, without EOS termination; not task accuracy.'}
        (args.output / 'result.json').write_text(json.dumps(result, indent=2) + '\n')
        print(json.dumps({k: v for k, v in result.items() if k not in ('actual', 'expected')}), flush=True)
        assert result['exact_token_parity'], 'Rolling decode changed token outputs.'


if __name__ == '__main__':
    main()
