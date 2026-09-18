import json
import os
import string
import time
from functools import partial
from itertools import product
from math import prod
from pathlib import Path

import torch
from fla.ops.gated_delta_rule import chunk_gated_delta_rule, fused_recurrent_gated_delta_rule
from transformers import AutoTokenizer, GenerationConfig, Qwen3_5ForConditionalGeneration
from transformers.models.qwen3_5 import modeling_qwen3_5

from jev_spawn.schema import BENCHMARK, CONTROLLER, controller_prompts, joint_controller_prompts
from jev_spawn.structured import score_fields


STOCK_KERNELS = (modeling_qwen3_5.torch_chunk_gated_delta_rule, modeling_qwen3_5.torch_recurrent_gated_delta_rule)


def unique_object(pairs):
    result = dict(pairs)
    if len(result) != len(pairs):
        raise ValueError('Generated JSON contains duplicate keys.')
    return result


def parse_choices(text, fields):
    result = json.loads(text, object_pairs_hook=unique_object)
    if not isinstance(result, dict) or set(result) != set(fields):
        raise ValueError('Generated JSON must contain exactly the requested field keys.')
    if any(not isinstance(result[name], str) or result[name] not in labels for name, labels in fields.items()):
        raise ValueError('Generated JSON contains an invalid option letter.')
    return result


def parse_array_choices(text, fields):
    result = json.loads(text)
    if not isinstance(result, list) or len(result) != len(fields):
        raise ValueError('Generated JSON must have exactly one array element per field.')
    if any(not isinstance(choice, str) or choice not in labels for choice, labels in zip(result, fields.values())):
        raise ValueError('Generated JSON array contains an invalid option letter.')
    return dict(zip(fields, result))


class FiniteOutputs:
    name = 'finite_json_token_trie'

    def __init__(self, tokenizer, outputs, eos_ids, prompt_width, max_new_tokens):
        sequences = tokenizer(outputs, add_special_tokens=False)['input_ids']
        assert outputs and tokenizer.batch_decode(sequences) == outputs
        assert max(map(len, sequences)) + 1 <= max_new_tokens, 'Token budget cannot complete every legal JSON output plus EOS.'
        self.root, self.eos_ids, self.prompt_width = {}, eos_ids, prompt_width
        self.output_count = len(outputs)
        for sequence in sequences:
            node = self.root
            for token in sequence:
                node = node.setdefault(token, {})
            for eos in eos_ids:
                node[eos] = {}

    def __call__(self, batch_id, input_ids):
        node = self.root
        for token in input_ids[self.prompt_width:].tolist():
            if token in self.eos_ids:
                return self.eos_ids
            node = node[token]
        return list(node)


class ArrayOutputs:
    name = 'ordered_json_array_automaton'

    def __init__(self, tokenizer, eos_ids, prompt_width, max_new_tokens, labels):
        assert labels and all(labels)
        letters = list(dict.fromkeys(letter for options in labels for letter in options))
        parts = ['["', '","', '"]', *letters]
        tokenized = tokenizer(parts, add_special_tokens=False)['input_ids']
        assert tokenizer.batch_decode(tokenized) == parts
        tokens = dict(zip(parts, tokenized))
        assert all(len(tokens[letter]) == 1 for letter in letters), 'Array labels must be single native tokens.'
        texts = [left + letter + right for left in parts[:2] for letter in letters for right in parts[1:3]]
        expected = [tokens[left] + tokens[letter] + tokens[right]
                    for left in parts[:2] for letter in letters for right in parts[1:3]]
        assert tokenizer(texts, add_special_tokens=False)['input_ids'] == expected, 'JSON concatenation changes native token boundaries.'
        required = len(tokens['["']) + len(labels) + (len(labels) - 1) * len(tokens['","']) + len(tokens['"]']) + 1
        assert required <= max_new_tokens, f'Array JSON needs {required} tokens including EOS.'
        self.nodes, self.eos_ids, self.prompt_width = [{}], eos_ids, prompt_width
        self.output_count = prod(map(len, labels))
        for slot, options in enumerate(labels):
            for token in tokens['["' if slot == 0 else '","']:
                self.nodes[-1][token] = len(self.nodes)
                self.nodes.append({})
            self.nodes[-1].update({tokens[letter][0]: len(self.nodes) for letter in options})
            self.nodes.append({})
        for token in tokens['"]']:
            self.nodes[-1][token] = len(self.nodes)
            self.nodes.append({})
        self.nodes[-1].update({eos: len(self.nodes) for eos in eos_ids})

    def __call__(self, batch_id, input_ids):
        node = 0
        for token in input_ids[self.prompt_width:].tolist():
            if token in self.eos_ids:
                return self.eos_ids
            node = self.nodes[node][token]
        return list(self.nodes[node])


def set_kernel(name):
    kernels = {"torch": STOCK_KERNELS, "fla": (chunk_gated_delta_rule, fused_recurrent_gated_delta_rule)}
    modeling_qwen3_5.torch_chunk_gated_delta_rule, modeling_qwen3_5.torch_recurrent_gated_delta_rule = kernels[name]


class Backend:
    def __init__(self, config):
        set_kernel(config["kernel"])
        self.config = config
        torch.set_num_threads(config["cpu_threads"])
        torch.manual_seed(config["seed"])
        self.tokenizer = AutoTokenizer.from_pretrained(
            config["model_path"], padding_side="left", local_files_only=True
        )
        if self.tokenizer.pad_token_id is None:
            raise ValueError("The model tokenizer must declare a padding token.")
        self.model = Qwen3_5ForConditionalGeneration.from_pretrained(
            config["model_path"],
            dtype=getattr(torch, config["dtype"]),
            attn_implementation=config["attention"],
            device_map="cuda:0",
            local_files_only=True,
        ).eval()
        self.device = torch.device("cuda:0")
        eos = self.model.generation_config.eos_token_id
        self.eos_ids = [eos] if isinstance(eos, int) else eos
        self.metadata = {
            "model_path": config["model_path"],
            "model_class": type(self.model).__name__,
            "parameters": sum(p.numel() for p in self.model.parameters()),
            "dtype": config["dtype"],
            "attention": config["attention"],
            "kernel": config["kernel"],
            "device": torch.cuda.get_device_name(self.device),
            "controller": "restricted next-token softmax; not calibrated",
        }

    def _render(self, prompts, system):
        return self.tokenizer.apply_chat_template(
            [[{"role": "system", "content": system}, {"role": "user", "content": p}]
             for p in prompts],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )

    def _encode(self, rendered):
        encoded = self.tokenizer(
            rendered, padding=True, truncation=False, add_special_tokens=False,
            return_tensors="pt",
        )
        lengths = encoded["attention_mask"].sum(dim=1).tolist()
        if max(lengths) > self.config["max_input_tokens"]:
            raise ValueError(
                f"Input has {max(lengths)} tokens; limit is "
                f"{self.config['max_input_tokens']}. Inputs are never truncated."
            )
        return encoded.to(self.device), lengths

    @torch.inference_mode()
    def generate(self, prompts, system, max_new_tokens, allowed_outputs=None, grammar_factory=None):
        assert allowed_outputs is None or grammar_factory is None
        torch.cuda.synchronize(self.device)
        started = time.perf_counter()
        inputs, lengths = self._encode(self._render(prompts, system))
        config = GenerationConfig(
            do_sample=False, max_new_tokens=max_new_tokens, use_cache=True,
            eos_token_id=self.eos_ids, pad_token_id=self.tokenizer.pad_token_id,
            bos_token_id=self.tokenizer.bos_token_id,
        )
        grammar = None
        if allowed_outputs is not None:
            grammar = FiniteOutputs(self.tokenizer, allowed_outputs, self.eos_ids,
                                    inputs['input_ids'].shape[1], max_new_tokens)
        if grammar_factory is not None:
            grammar = grammar_factory(self.tokenizer, self.eos_ids, inputs['input_ids'].shape[1], max_new_tokens)
        sequences = self.model.generate(
            **inputs, generation_config=config, logits_to_keep=1, prefix_allowed_tokens_fn=grammar,
        )
        generated = sequences[:, inputs["input_ids"].shape[1]:]
        eos_mask = torch.isin(generated, torch.tensor(self.eos_ids, device=self.device))
        finished = eos_mask.any(dim=1)
        first_eos = eos_mask.int().argmax(dim=1) + 1
        counts = torch.where(finished, first_eos, generated.shape[1]).tolist()
        texts = self.tokenizer.batch_decode(generated, skip_special_tokens=True)
        torch.cuda.synchronize(self.device)
        return {
            "texts": texts,
            "input_tokens": lengths,
            "output_tokens": counts,
            "elapsed_seconds": time.perf_counter() - started,
            "truncated": (~finished).tolist(),
            "batch_size": len(prompts),
            "grammar": grammar.name if grammar is not None else "none",
            "allowed_output_count": grammar.output_count if grammar is not None else None,
            "peak_cuda_memory_bytes": torch.cuda.max_memory_allocated(self.device),
        }

    @torch.inference_mode()
    def score(self, states, question, options):
        if not 2 <= len(options) <= len(string.ascii_uppercase):
            raise ValueError("The controller requires between 2 and 26 options.")
        torch.cuda.synchronize(self.device)
        started = time.perf_counter()
        labels = list(string.ascii_uppercase[:len(options)])
        prompts = controller_prompts(
            states, question, options, labels, CONTROLLER["output_instruction"]
        )
        rendered = self._render(prompts, CONTROLLER["system"])
        inputs, lengths = self._encode(rendered)
        label_ids = self.tokenizer(labels, add_special_tokens=False)["input_ids"]
        if any(len(ids) != 1 for ids in label_ids):
            raise ValueError("Every option label must encode as one token.")
        # Verify the actual answer boundary, not just isolated label tokenization.
        prefixes = self.tokenizer(rendered, add_special_tokens=False)["input_ids"]
        joined = self.tokenizer(
            [prefix + label for prefix in rendered for label in labels],
            add_special_tokens=False,
        )["input_ids"]
        expected = [prefix + ids for prefix in prefixes for ids in label_ids]
        if joined != expected:
            raise ValueError("Option labels are not single tokens at the answer boundary.")
        ids = [row[0] for row in label_ids]
        logits = self.model(**inputs, use_cache=False, logits_to_keep=1).logits[:, -1, ids]
        probabilities = logits.float().softmax(dim=-1)
        choices = probabilities.argmax(dim=-1).tolist()
        values = probabilities.tolist()
        torch.cuda.synchronize(self.device)
        return {
            "choices": [options[index]["id"] for index in choices],
            "probabilities": values,
            "input_tokens": lengths,
            "elapsed_seconds": time.perf_counter() - started,
            "batch_size": len(states),
            "option_ids": [option["id"] for option in options],
            "peak_cuda_memory_bytes": torch.cuda.max_memory_allocated(self.device),
        }

    def score_fields(self, states, fields, mode='shared'):
        return score_fields(self, states, fields, mode)

    def _parse_json(self, result, labels, context, parser=parse_choices):
        try:
            return [parser(text, labels) for text in result['texts']]
        except ValueError as error:
            directory = Path(self.config['run_dir'])
            directory.mkdir(parents=True, exist_ok=True)
            with (directory / f'controller-failures-{os.getpid()}.jsonl').open('a') as stream:
                stream.write(json.dumps({'timestamp': time.time(), 'error': str(error),
                                         'context': context, 'result': result}, ensure_ascii=False) + '\n')
            raise

    def _json_outputs(self, labels):
        constraints = self.config['json_constraints']
        assert constraints in {'finite', 'none'}
        if constraints == 'none':
            return None
        return [json.dumps(dict(zip(labels, values)), separators=(',', ':'))
                for values in product(*labels.values())]

    def generate_fields(self, states, fields, max_new_tokens):
        assert states and fields and max_new_tokens > 0
        assert all(2 <= len(field['options']) <= len(string.ascii_uppercase) for field in fields.values())
        started = time.perf_counter()
        labels = {name: list(string.ascii_uppercase[:len(field['options'])]) for name, field in fields.items()}
        prompts = joint_controller_prompts(states, fields, string.ascii_uppercase)
        result = self.generate(prompts, CONTROLLER['system'], max_new_tokens,
                               allowed_outputs=self._json_outputs(labels))
        parsed = self._parse_json(result, labels, {'states': states, 'fields': fields})
        return self._field_results(result, parsed, fields, labels, 'joint_json', started)

    def generate_array_fields(self, states, fields, max_new_tokens):
        assert states and fields and max_new_tokens > 0
        assert self.config['json_constraints'] == 'finite'
        schemas = fields if isinstance(fields, list) else [fields] * len(states)
        assert len(schemas) == len(states)
        schema = schemas[0]
        signature = lambda row: tuple((name, tuple(option['id'] for option in field['options'])) for name, field in row.items())
        assert all(signature(row) == signature(schema) for row in schemas), 'Each array batch needs the same ordered field and option IDs.'
        assert all(2 <= len(field['options']) <= len(string.ascii_uppercase) for field in schema.values())
        started = time.perf_counter()
        labels = {name: list(string.ascii_uppercase[:len(field['options'])]) for name, field in schema.items()}
        prompts = [joint_controller_prompts([state], row, string.ascii_uppercase, BENCHMARK['array_output_instruction'])[0]
                   for state, row in zip(states, schemas)]
        result = self.generate(prompts, CONTROLLER['system'], max_new_tokens,
                               grammar_factory=partial(ArrayOutputs, labels=list(labels.values())))
        parsed = self._parse_json(result, labels, {'states': states, 'fields': fields}, parser=parse_array_choices)
        return self._field_results(result, parsed, schema, labels, 'compact_json', started)

    def _field_results(self, result, parsed, fields, labels, mode, started):
        batch_size = result['batch_size']
        answers = {}
        for name, field in fields.items():
            answers[name] = {
                'choices': [field['options'][labels[name].index(row[name])]['id'] for row in parsed],
                'probabilities': [None] * batch_size, 'option_logits': [None] * batch_size,
                'option_ids': [option['id'] for option in field['options']],
                'input_tokens': result['input_tokens'], 'batch_size': batch_size,
            }
        elapsed = time.perf_counter() - started
        return {
            **result, 'fields': answers, 'mode': mode, 'field_count': len(fields),
            'elapsed_seconds': elapsed,
            'logical_input_tokens': sum(result['input_tokens']),
            'computed_input_tokens': sum(result['input_tokens']),
            'padded_input_tokens': batch_size * max(result['input_tokens']),
            'timings': {'generation_seconds': result['elapsed_seconds'],
                        'format_and_parse_seconds': elapsed - result['elapsed_seconds']},
        }

    def decide(self, states, question, options, mode, max_new_tokens):
        if mode == "direct":
            result = self.score(states, question, options)
            result["output_tokens"] = [0] * len(states)
            return result
        labels = list(string.ascii_uppercase[:len(options)])
        prompts = controller_prompts(states, question, options, labels, BENCHMARK["json_output_instruction"])
        result = self.generate(prompts, CONTROLLER["system"], max_new_tokens,
                               allowed_outputs=self._json_outputs({'choice': labels}))
        parsed = self._parse_json(result, {'choice': labels}, {'states': states, 'question': question, 'options': options})
        choices = [row['choice'] for row in parsed]
        result["choices"] = [options[labels.index(choice)]["id"] for choice in choices]
        result["probabilities"] = [None] * len(states)
        return result
