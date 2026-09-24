import random
import time

import numpy as np
import torch
import torch.distributed as dist
from fla.ops.gated_delta_rule import chunk_gated_delta_rule, fused_recurrent_gated_delta_rule
from transformers import AutoTokenizer, GenerationConfig, Qwen3_5ForConditionalGeneration
from transformers.models.qwen3_5 import modeling_qwen3_5

from jev_spawn.schema import CONTROLLER, controller_prompts
from jev_spawn.algo.structured import score_fields
from jev_spawn.infra.readout_labels import native_labels, validate_boundaries
from jev_spawn.infra.configuration import CORE, EXECUTION_POLICY, load_resource, resolve_symbol
from jev_spawn.infra.qwen35 import install_qwen35_execution
from jev_spawn.infra.qwen35.parallel import local_cache_config, shard_qwen35


KERNELS = {name: tuple(resolve_symbol(symbol) for symbol in symbols)
           for name, symbols in EXECUTION_POLICY['backend']['kernels'].items()}
STOCK_KERNELS = KERNELS['torch']


def set_kernel(name):
    modeling_qwen3_5.torch_chunk_gated_delta_rule, modeling_qwen3_5.torch_recurrent_gated_delta_rule = KERNELS[name]


class Backend:
    def selected_output_weights(self, token_ids):
        if self.config['world_size'] > 1:
            return self.model.lm_head.selected_weight(tuple(token_ids))
        indices = torch.tensor(token_ids, device=self.device)
        return self.model.lm_head.weight.index_select(0, indices)

    def __init__(self, config):
        set_kernel(config["kernel"])
        self.config = config
        torch.set_num_threads(config["cpu_threads"])
        random.seed(config["seed"])
        np.random.seed(config["seed"])
        torch.manual_seed(config["seed"])
        torch.cuda.manual_seed_all(config["seed"])
        self.tokenizer = AutoTokenizer.from_pretrained(
            config["model_path"], padding_side=CORE["backend"]["padding_side"], local_files_only=CORE["backend"]["local_files_only"]
        )
        self.answer_labels, self.answer_label_ids = native_labels(self.tokenizer, load_resource('readout_labels'))
        self.answer_boundary_cache = set()
        if self.tokenizer.pad_token_id is None:
            raise ValueError("The model tokenizer must declare a padding token.")
        self.device = torch.device(CORE['backend']['device'])
        if config['world_size'] > 1:
            assert dist.is_initialized() and dist.get_world_size() == config['world_size']
            assert config['execution'] == 'qwen35_optimized'
            self.device = torch.device('cuda', torch.cuda.current_device())
        self.model = Qwen3_5ForConditionalGeneration.from_pretrained(
            config["model_path"],
            dtype=getattr(torch, config["dtype"]),
            attn_implementation=config["attention"],
            device_map=str(self.device),
            local_files_only=CORE["backend"]["local_files_only"],
        ).eval()
        global_parameters = sum(parameter.numel() for parameter in self.model.parameters())
        self.vocab_size = self.model.config.get_text_config(decoder=True).vocab_size
        self.cache_config = self.model.config
        if config['world_size'] > 1:
            shard_qwen35(self.model, load_resource('qwen35_execution'), dist.group.WORLD)
            self.cache_config = local_cache_config(self.model, dist.group.WORLD)
        if config['execution'] == 'qwen35_optimized':
            install_qwen35_execution(self.model, load_resource('qwen35_execution'))
        with torch.inference_mode():
            self.finite_output_weights = self.selected_output_weights(
                tuple(self.answer_label_ids)).float().contiguous()
        eos = self.model.generation_config.eos_token_id
        self.eos_ids = [eos] if isinstance(eos, int) else eos
        self.metadata = {
            "model_path": config["model_path"],
            "model_class": type(self.model).__name__,
            "parameters": sum(p.numel() for p in self.model.parameters()),
            "global_parameters": global_parameters,
            "world_size": config['world_size'],
            "dtype": config["dtype"],
            "attention": config["attention"],
            "kernel": config["kernel"],
            "execution": config['execution'],
            "decode_attention_splits": config['decode_attention_splits'],
            "device": torch.cuda.get_device_name(self.device),
            "controller": "restricted next-token softmax; not calibrated",
            "core_config": CORE,
            "answer_labels": self.answer_labels,
            "answer_label_ids": self.answer_label_ids,
        }

    def _render(self, prompts, system):
        return self.tokenizer.apply_chat_template(
            [[{"role": "system", "content": system}, {"role": "user", "content": p}]
             for p in prompts],
            tokenize=False,
            add_generation_prompt=CORE["backend"]["add_generation_prompt"],
            enable_thinking=CORE["backend"]["enable_thinking"],
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
    def generate(self, prompts, system, max_new_tokens):
        torch.cuda.synchronize(self.device)
        started = time.perf_counter()
        inputs, lengths = self._encode(self._render(prompts, system))
        config = GenerationConfig(
            do_sample=CORE["backend"]["do_sample"], max_new_tokens=max_new_tokens,
            use_cache=CORE["backend"]["generation_use_cache"], eos_token_id=self.eos_ids,
            pad_token_id=self.tokenizer.pad_token_id, bos_token_id=self.tokenizer.bos_token_id,
        )
        sequences = self.model.generate(
            **inputs, generation_config=config, logits_to_keep=CORE["backend"]["logits_to_keep"],
        )
        generated = sequences[:, inputs["input_ids"].shape[1]:]
        eos_mask = torch.isin(generated, torch.tensor(self.eos_ids, device=self.device))
        finished = eos_mask.any(dim=1)
        first_eos = eos_mask.int().argmax(dim=1) + 1
        counts = torch.where(finished, first_eos, generated.shape[1]).tolist()
        texts = self.tokenizer.batch_decode(generated, skip_special_tokens=True)
        torch.cuda.synchronize(self.device)
        return {
            "texts": texts, "input_tokens": lengths, "output_tokens": counts,
            "elapsed_seconds": time.perf_counter() - started,
            "truncated": (~finished).tolist(), "batch_size": len(prompts),
            "peak_cuda_memory_bytes": torch.cuda.max_memory_allocated(self.device),
        }

    @torch.inference_mode()
    def score(self, states, question, options):
        if not CORE['controller']['minimum_options'] <= len(options) <= len(self.answer_labels):
            raise ValueError(f"The controller requires between {CORE['controller']['minimum_options']} "
                             f"and {len(self.answer_labels)} options.")
        torch.cuda.synchronize(self.device)
        started = time.perf_counter()
        labels = list(self.answer_labels[:len(options)])
        prompts = controller_prompts(
            states, question, options, labels, CONTROLLER["output_instruction"]
        )
        rendered = self._render(prompts, CONTROLLER["system"])
        inputs, lengths = self._encode(rendered)
        ids = self.answer_label_ids[:len(options)]
        prefixes = self.tokenizer(rendered, add_special_tokens=False)['input_ids']
        validate_boundaries(self.tokenizer, rendered, prefixes, labels, ids, [len(options)] * len(rendered))
        logits = self.model(**inputs, use_cache=False, logits_to_keep=CORE["backend"]["logits_to_keep"]).logits[:, -1, ids]
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

    def score_fields(self, states, fields, mode):
        return score_fields(self, states, fields, mode)
