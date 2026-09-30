from dataclasses import asdict, dataclass
from pathlib import Path

import yaml


@dataclass(frozen=True)
class Model:
    path: str
    dtype: str
    attention: str
    kernel: str
    max_input_tokens: int
    execution: str
    decode_attention_splits: int

@dataclass(frozen=True)
class Runtime:
    batch_size: int
    root_batch_size: int
    branch_batch_size: int
    batch_wait_seconds: float
    cpu_threads: int
    world_size: int
    seed: int
    sample_timeout_seconds: float | None
    max_turns: int
    decode_engine: str
    generation_scheduling: str
    row_delivery: str
    stop_engine: str
    graph_cache_size: int
    graph_warmup_steps: int
    graph_cache_block_tokens: int
    cache_allocation: str


@dataclass(frozen=True)
class Generation:
    enable_thinking: bool
    max_new_tokens: int
    temperature: float


@dataclass(frozen=True)
class SharedConfig:
    model: Model
    runtime: Runtime
    generation: Generation

    @classmethod
    def load(cls, path):
        data = yaml.safe_load(Path(path).read_text())
        assert set(data) == {'model', 'runtime', 'generation'}
        config = cls(Model(**data['model']), Runtime(**data['runtime']),
                     Generation(**data['generation']))
        assert config.model.dtype == 'bfloat16' and config.runtime.world_size > 0
        if config.runtime.world_size > 1:
            assert config.model.execution == 'qwen35_optimized'
            assert config.runtime.decode_engine == 'cuda_graph'
        assert config.model.execution in {'native', 'qwen35_optimized'}
        assert config.model.decode_attention_splits >= 0
        assert min(config.runtime.batch_size, config.runtime.root_batch_size,
                   config.runtime.max_turns,
                   config.runtime.branch_batch_size, config.runtime.cpu_threads,
                   config.model.max_input_tokens, config.generation.max_new_tokens) > 0
        assert config.runtime.sample_timeout_seconds is None or config.runtime.sample_timeout_seconds > 0
        assert config.runtime.batch_wait_seconds >= 0
        assert config.runtime.decode_engine in {'eager', 'cuda_graph'}
        assert config.runtime.generation_scheduling == 'cohort'
        assert config.runtime.row_delivery in {'cohort', 'immediate'}
        assert config.runtime.stop_engine in {'hf', 'indexed'}
        assert config.runtime.row_delivery != 'immediate' or config.runtime.decode_engine == 'cuda_graph'
        assert config.runtime.graph_cache_size > 0 and config.runtime.graph_warmup_steps > 0
        assert config.runtime.graph_cache_block_tokens > 0
        assert config.runtime.cache_allocation in {'independent_static_cache_v1', 'shared_static_cache_arena_v1'}
        if config.runtime.cache_allocation == 'shared_static_cache_arena_v1':
            assert config.runtime.decode_engine == 'cuda_graph'
            assert config.runtime.generation_scheduling == 'cohort'
        assert config.generation.temperature >= 0
        assert config.generation.enable_thinking is False
        return config

    def method_settings(self, method):
        shared = {**asdict(self.model), **asdict(self.runtime), **asdict(self.generation),
                  'context_length': self.model.max_input_tokens,
                  'model_path': self.model.path}
        overlap = method.keys() & shared.keys()
        if overlap:
            raise ValueError(f'Method settings redefine shared parameters: {sorted(overlap)}')
        return {**method, **shared}

    def backend(self):
        return dict(model_path=self.model.path, dtype=self.model.dtype,
                    attention=self.model.attention, kernel=self.model.kernel,
                    execution=self.model.execution, decode_attention_splits=self.model.decode_attention_splits,
                    max_input_tokens=self.model.max_input_tokens, seed=self.runtime.seed,
                    cpu_threads=self.runtime.cpu_threads, world_size=self.runtime.world_size,
                    batch_size=self.runtime.batch_size, branch_batch_size=self.runtime.branch_batch_size)
