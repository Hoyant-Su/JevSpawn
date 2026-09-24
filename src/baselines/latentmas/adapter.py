import ast
import importlib.util
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Optional, Tuple

import torch
from transformers.cache_utils import LinearAttentionCacheLayerMixin

from baselines.latentmas.parallel_alignment import distributed_alignment


SOURCE = Path(__file__).resolve().parents[3] / "external/LatentMAS"


def source_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


PROMPTS = source_module("latentmas_original_prompts", SOURCE / "prompts.py")
UTILS = source_module("latentmas_original_utils", SOURCE / "utils.py")
AGENTS = source_module("latentmas_original_agents", SOURCE / "methods/__init__.py")


def task_messages(**kwargs):
    messages = PROMPTS.build_agent_message_sequential_latent_mas(**kwargs)
    choices = kwargs["args"].answer_options[kwargs["question"]]
    return [{**m, "content": m["content"].replace("A,B,C,D.", ",".join(choices) + ".")}
            for m in messages]


def task_item(task):
    options = task["fields"]["q0"]["options"]
    return {"question": task["state"] + "\n" + "\n".join(
        option["id"] + ". " + option["description"] for option in options), "gold": "", "solution": ""}


def original_class(path, name, methods, namespace):
    tree = ast.parse(path.read_text(), filename=str(path))
    original = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == name)
    original.body = [node for node in original.body if isinstance(node, ast.FunctionDef) and node.name in methods]
    assert {node.name for node in original.body} == set(methods)
    selected = ast.Module(body=[original], type_ignores=[])
    exec(compile(selected, str(path), "exec"), namespace)
    return namespace[name]


NAMESPACE = dict(torch=torch, Dict=Dict, List=List, Optional=Optional, Tuple=Tuple,
                 default_agents=AGENTS.default_agents,
                 _past_length=lambda cache: 0 if cache is None else cache.get_seq_length(),
                 build_agent_message_sequential_latent_mas=task_messages,
                 extract_gsm8k_answer=UTILS.extract_gsm8k_answer,
                 normalize_answer=UTILS.normalize_answer)
OriginalWrapper = original_class(SOURCE / "models.py", "ModelWrapper", [
    "prepare_chat_batch", "_build_latent_realign_matrix", "_ensure_latent_realign_matrix",
    "_apply_latent_realignment", "generate_latent_batch"], NAMESPACE)
OriginalMethod = original_class(SOURCE / "methods/latent_mas.py", "LatentMASMethod",
                                ["run_batch"], NAMESPACE)


class HybridTransport:
    def __init__(self, backend):
        self.backend = backend
        self.trunk = backend.model.model.language_model
        self.mask = None
        self.cache = None
        self.phase = "prefill"
        self.records = []
        self.last_hidden = None
        self.restored_rows = 0

    def get_input_embeddings(self):
        return self.backend.model.get_input_embeddings()

    def get_output_embeddings(self):
        return self.backend.model.get_output_embeddings()

    def _forward(self, embeddings, mask, cache):
        history = mask if self.mask is None else torch.cat([self.mask, mask], dim=1)
        assert history.shape[1] <= self.backend.config["max_input_tokens"]
        positions = (history.cumsum(1) - 1).clamp_min(0)[:, -embeddings.shape[1]:]
        inactive = ~mask[:, 0].bool()
        restore = cache is not None and embeddings.shape[1] == 1 and bool(inactive.any())
        saved = []
        if restore:
            for layer in cache.layers:
                if isinstance(layer, LinearAttentionCacheLayerMixin):
                    saved.append((layer, layer.conv_states[0][inactive].clone(),
                                  layer.recurrent_states[0][inactive].clone()))
        torch.cuda.synchronize()
        start = time.perf_counter()
        output = self.trunk(inputs_embeds=embeddings, attention_mask=history,
                            position_ids=positions, past_key_values=cache,
                            use_cache=True, output_hidden_states=True, return_dict=True)
        if restore:
            for layer, conv, recurrent in saved:
                layer.conv_states[0][inactive] = conv
                layer.recurrent_states[0][inactive] = recurrent
                assert torch.equal(layer.conv_states[0][inactive], conv)
                assert torch.equal(layer.recurrent_states[0][inactive], recurrent)
            self.restored_rows += int(inactive.sum())
        torch.cuda.synchronize()
        self.records.append({"phase": self.phase, "batch_size": embeddings.shape[0],
                             "tokens": embeddings.shape[1], "seconds": time.perf_counter() - start,
                             "valid_tokens": int(mask.sum()), "restored_rows": int(inactive.sum()) if restore else 0})
        self.mask, self.cache = history, output.past_key_values
        self.last_hidden = output.last_hidden_state[:, -1]
        return output

    def __call__(self, input_ids=None, inputs_embeds=None, attention_mask=None,
                 past_key_values=None, **kwargs):
        if past_key_values is None:
            self.mask, self.cache = None, None
        else:
            assert past_key_values is self.cache
        embeddings = self.get_input_embeddings()(input_ids) if input_ids is not None else inputs_embeds
        local_mask = attention_mask[:, -embeddings.shape[1]:]
        # Interior padding must not advance either recurrent state or convolution history.
        padding_prefix = (int((local_mask == 0).sum(1).max())
                          if past_key_values is not None and embeddings.shape[1] > 1 else 0)
        for offset in range(padding_prefix):
            self._forward(embeddings[:, offset:offset + 1], local_mask[:, offset:offset + 1], self.cache)
        return self._forward(embeddings[:, padding_prefix:], local_mask[:, padding_prefix:], self.cache)

    def snapshot(self):
        result = {"mask": self.mask.detach().cpu(), "hidden": self.last_hidden.detach().cpu(), "layers": []}
        for layer in self.cache.layers:
            if isinstance(layer, LinearAttentionCacheLayerMixin):
                result["layers"].append({"conv": layer.conv_states[0].detach().cpu().clone(),
                                         "recurrent": layer.recurrent_states[0].detach().cpu().clone()})
            else:
                result["layers"].append({"keys": layer.keys.detach().cpu().clone(),
                                         "values": layer.values.detach().cpu().clone()})
        return result


class ModelAdapter(OriginalWrapper):
    def __init__(self, backend, args):
        self.backend, self.args = backend, args
        self.model = HybridTransport(backend)
        self.tokenizer, self.device = backend.tokenizer, backend.device
        self._latent_realign_matrices = {}
        self.pre_aligned = None
        self.capture = False
        self.snapshots, self.output_ids, self.itl = [], [], []
        self.completion_times = []
        self.role_times = []

    def _build_latent_realign_matrix(self, model, device, args):
        if self.backend.config['world_size'] > 1:
            return distributed_alignment(model, device, args)
        return super()._build_latent_realign_matrix(model, device, args)

    def render_chat(self, messages, add_generation_prompt=True):
        return self.tokenizer.apply_chat_template(messages, tokenize=False,
                                                  add_generation_prompt=add_generation_prompt,
                                                  enable_thinking=False)

    def generate_latent_batch(self, input_ids, attention_mask, *, latent_steps, past_key_values):
        start = time.perf_counter()
        self.model.phase = "latent_role"
        cache = super().generate_latent_batch(input_ids, attention_mask,
                                             latent_steps=latent_steps, past_key_values=past_key_values)
        torch.cuda.synchronize()
        self.role_times.append(time.perf_counter() - start)
        if self.capture:
            self.snapshots.append(self.model.snapshot())
        return cache

    def generate_text_batch(self, input_ids, attention_mask, *, max_new_tokens,
                            temperature, top_p, past_key_values):
        assert temperature == 0
        start = time.perf_counter()
        self.model.phase = "judger_prefill"
        output = self.model(input_ids=input_ids, attention_mask=attention_mask,
                            past_key_values=past_key_values)
        if self.capture:
            self.snapshots.append(self.model.snapshot())
        self.model.phase = "judger_decode"
        finished = torch.zeros(input_ids.shape[0], dtype=torch.bool, device=self.device)
        tokens, completion_times = [], []
        for step in range(max_new_tokens):
            logits = self.backend.model.lm_head(output.last_hidden_state[:, -1])
            next_ids = logits.argmax(-1)
            next_ids = torch.where(finished, self.tokenizer.pad_token_id, next_ids)
            tokens.append(next_ids)
            torch.cuda.synchronize()
            completion_times.append(time.perf_counter())
            finished |= torch.isin(next_ids, torch.tensor(self.backend.eos_ids, device=self.device))
            if bool(finished.all()) or step + 1 == max_new_tokens:
                break
            output = self.model(input_ids=next_ids[:, None], attention_mask=(~finished).long()[:, None],
                                past_key_values=self.model.cache)
        sequences = torch.stack(tokens, dim=1).tolist()
        self.completion_times = completion_times
        self.output_ids = []
        self.itl = []
        for sequence in sequences:
            stop = next((i + 1 for i, token in enumerate(sequence) if token in self.backend.eos_ids), len(sequence))
            self.output_ids.append(sequence[:stop])
            self.itl.append([completion_times[i] - completion_times[i - 1] for i in range(1, stop)])
        self.role_times.append(time.perf_counter() - start)
        return self.tokenizer.batch_decode(self.output_ids, skip_special_tokens=True), self.model.cache

    def reset(self, capture=False):
        self.capture = capture
        self.snapshots, self.output_ids, self.itl, self.role_times = [], [], [], []
        self.completion_times = []
        self.model.records, self.model.restored_rows = [], 0
        self.model.cache, self.model.mask = None, None


def method(backend, config, tasks):
    assert config["memory_mode"] == "full" and config["alignment"]
    assert config["final_temperature"] == 0
    args = SimpleNamespace(prompt="sequential", think=False, task="arc_challenge",
                           model_name=backend.config["model_path"], latent_space_realign=True,
                           answer_options={task_item(task)["question"]: [option["id"] for option in task["fields"]["q0"]["options"]]
                                           for task in tasks})
    wrapper = ModelAdapter(backend, args)
    instance = OriginalMethod()
    instance.args, instance.model = args, wrapper
    instance.generate_bs = config["root_batch_size"]
    instance.latent_steps = config["latent_steps"]
    instance.judger_max_new_tokens = config["final_tokens"]
    instance.temperature, instance.top_p = config["final_temperature"], 1
    instance.agents = AGENTS.default_agents()
    instance.method_name, instance.task = "latent_mas", args.task
    instance.sequential_info_only = instance.latent_only = False
    return instance
