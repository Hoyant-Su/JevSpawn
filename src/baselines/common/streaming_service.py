from concurrent.futures import Future
from copy import deepcopy
from queue import Queue
import time

import torch
from transformers import TextStreamer

from baselines.common.service import BatchedRequest, BatchService


class StreamRequest(BatchedRequest):
    pass


class StreamText(TextStreamer):
    def __init__(self, tokenizer, stops, channel):
        super().__init__(tokenizer, skip_prompt=False, skip_special_tokens=True)
        self.stops, self.channel = stops, channel
        self.pending, self.closed = '', False

    def on_finalized_text(self, text, stream_end=False):
        if self.closed:
            return
        self.pending += text
        positions = [self.pending.find(stop) for stop in self.stops if stop in self.pending]
        if positions:
            text, self.pending = self.pending[:min(positions)], ''
            self.closed = True
        else:
            held = max((len(stop) - 1 for stop in self.stops), default=0)
            boundary = len(self.pending) if stream_end else max(0, len(self.pending) - held)
            text, self.pending = self.pending[:boundary], self.pending[boundary:]
        if text:
            self.channel.put((text, time.perf_counter()))


class StreamingStops:
    def __init__(self, stopping, batch, channels, tokenizer):
        self.stopping = stopping
        self.streams = {index: StreamText(tokenizer, request.stop, channels[id(request)])
                        for index, request in enumerate(batch) if isinstance(request, StreamRequest)}
        self.completed = set()

    def __getattr__(self, name):
        return getattr(self.stopping, name)

    def __call__(self, input_ids, scores, **kwargs):
        done = self.stopping(input_ids, scores, **kwargs)
        rows = [index for index in self.streams if index not in self.completed]
        if rows:
            indices = torch.tensor(rows, device=input_ids.device)
            tokens = input_ids[:, -1].index_select(0, indices).tolist()
            finished = done.index_select(0, indices).tolist()
            for index, token, ended in zip(rows, tokens, finished, strict=True):
                self.streams[index].put(torch.tensor([token]))
                if ended:
                    self.streams[index].end()
                    self.completed.add(index)
        return done


class StreamingBatchService(BatchService):
    def __init__(self, backend, shared, deadlines, *, settings, prompts):
        self.stream_channels = {}
        super().__init__(backend, shared, deadlines, settings=settings, prompts=prompts)

    def open_stream(self, messages, max_tokens, temperature, stop, *, task_id):
        assert 0 < max_tokens <= self.shared.generation.max_new_tokens
        assert temperature == self.shared.generation.temperature
        self.deadlines.remaining(task_id)
        stops = (stop,) if isinstance(stop, str) else tuple(stop or ())
        request = StreamRequest(deepcopy(messages), max_tokens, temperature, stops, task_id,
                                time.perf_counter(), Future())
        channel = Queue()
        self.stream_channels[id(request)] = channel
        request.future.add_done_callback(lambda future: channel.put((future, time.perf_counter())))
        self.requests.put(request)
        return request, channel

    def _stopping(self, batch, prompt_width):
        stopping = super()._stopping(batch, prompt_width)
        leader = self.shared.runtime.world_size == 1 or self.parallel_commands.is_leader
        if leader and any(isinstance(request, StreamRequest) for request in batch):
            stopping = StreamingStops(stopping, batch, self.stream_channels, self.backend.tokenizer)
            self.stopping = stopping
        return stopping

    def release_stream(self, request):
        self.stream_channels.pop(id(request))
