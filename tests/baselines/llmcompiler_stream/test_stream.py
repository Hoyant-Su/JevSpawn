import asyncio
from concurrent.futures import Future
from functools import partial
import json
from pathlib import Path
import pickle
from queue import Queue
from types import SimpleNamespace
import time
import unittest

import torch
from transformers import AutoTokenizer

from baselines.common.llmcompiler_stream import SchemaStreamingPlanner, StreamingCallbackLLM
from baselines.common.parallel_service import decode_request, encode_request, parallel_factory
from baselines.common.streaming_service import StreamRequest, StreamText, StreamingBatchService, StreamingStops
from src.llm_compiler.planner import Planner
from src.llm_compiler.task_fetching_unit import TaskFetchingUnit
from src.tools.base import Tool


CONFIG = json.loads(Path(__file__).with_name('config.json').read_text())
EVIDENCE = {}


class TestChannelService:
    def __init__(self, tokenizer, tool_started, plan, first_line):
        self.tokenizer, self.tool_started = tokenizer, tool_started
        self.deadlines = SimpleNamespace(remaining=lambda task: CONFIG['timeout_seconds'])
        self.plan, self.first_line = plan, first_line
        self.producer = None
        self.finished = False
        self.released = False

    def complete(self, *args, **kwargs):
        raise AssertionError('A streaming planner must not call a completed-response transport.')

    def open_stream(self, messages, max_tokens, temperature, stop, *, task_id):
        request = StreamRequest(messages, max_tokens, temperature, tuple(stop), task_id,
                                CONFIG['temperature'], Future())
        channel = Queue()
        request.future.add_done_callback(lambda future: channel.put((future, time.perf_counter())))
        self.producer = asyncio.create_task(self.produce(channel, request))
        return request, channel

    async def produce(self, channel, request):
        streamer = StreamText(self.tokenizer, request.stop, channel)
        # The first complete action must actually execute before remaining plan tokens exist.
        first = self.first_line
        for token in self.tokenizer.encode(first, add_special_tokens=False):
            streamer.put(torch.tensor([token]))
        await asyncio.wait_for(self.tool_started.wait(), CONFIG['timeout_seconds'])
        remainder = self.plan[len(first):]
        for token in self.tokenizer.encode(remainder, add_special_tokens=False):
            streamer.put(torch.tensor([token]))
        streamer.end()
        self.finished = True
        request.future.set_result(self.plan)

    def release_stream(self, request):
        self.released = True


class StreamingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tokenizer = AutoTokenizer.from_pretrained(CONFIG['tokenizer'], local_files_only=True)

    def test_stop_text_and_unicode_exact(self):
        for text in CONFIG['texts']:
            channel = Queue()
            streamer = StreamText(self.tokenizer, (CONFIG['stop'],), channel)
            tokens = self.tokenizer.encode(text, add_special_tokens=False)
            for token in tokens:
                streamer.put(torch.tensor([token]))
            streamer.end()
            fragments = []
            while not channel.empty():
                fragments.append(channel.get_nowait()[0])
            self.assertEqual(''.join(fragments), text.split(CONFIG['stop'])[0])
        EVIDENCE['unicode_and_stop_boundaries_exact'] = True

    def test_variable_rows_complete_once(self):
        texts = [text.split(CONFIG['stop'])[0] for text in CONFIG['texts']]
        tokens = [self.tokenizer.encode(text, add_special_tokens=False) for text in texts]
        batch = [StreamRequest([], CONFIG['max_tokens'], CONFIG['temperature'], (), str(index),
                 CONFIG['temperature'], Future()) for index in range(len(tokens))]
        channels = {id(item): Queue() for item in batch}
        lengths = torch.tensor([len(row) for row in tokens])
        stopping = StreamingStops(lambda ids, scores: lengths <= ids.shape[1], batch, channels, self.tokenizer)
        for step in range(max(map(len, tokens))):
            inputs = torch.tensor([row[:step + 1] + [self.tokenizer.eos_token_id] * max(0, step + 1 - len(row))
                                   for row in tokens])
            done = stopping(inputs, None)
            self.assertTrue(torch.equal(done, lengths <= step + 1))
        for item, text in zip(batch, texts, strict=True):
            channel, pieces = channels[id(item)], []
            while not channel.empty():
                pieces.append(channel.get_nowait()[0])
            self.assertEqual(''.join(pieces), text)
        EVIDENCE['variable_length_rows_preserved'] = True

    def test_tp_request_roundtrip(self):
        item = StreamRequest([], CONFIG['max_tokens'], CONFIG['temperature'], (CONFIG['stop'],),
                             CONFIG['task_id'], CONFIG['temperature'], Future())
        item.input_ids = self.tokenizer.encode(CONFIG['plan'], add_special_tokens=False)
        restored = decode_request(pickle.loads(pickle.dumps(encode_request(item))))
        self.assertIs(type(restored), StreamRequest)
        self.assertEqual(encode_request(restored), encode_request(item))
        self.assertTrue(issubclass(parallel_factory(StreamingBatchService), StreamingBatchService))

    def test_original_planner_dispatch_before_completion(self):
        self.check_dispatch(Planner, CONFIG['plan'], CONFIG['first_line'])

    def test_schema_stream_parser_preserves_keyword_arguments(self):
        self.check_dispatch(SchemaStreamingPlanner, CONFIG['keyword_plan'], CONFIG['keyword_first_line'])

    def check_dispatch(self, planner_type, plan, first_line):
        async def run():
            started = asyncio.Event()
            service = TestChannelService(self.tokenizer, started, plan, first_line)
            inputs, dispatch = [], []

            async def invoke(value, *, fields):
                inputs.append(value)
                dispatch.append(service.finished)
                started.set()
                return CONFIG['first_result'] if value == CONFIG['expected_tool_inputs'][0] else CONFIG['second_result']

            tools = [Tool(name=CONFIG['tool_name'], func=partial(invoke, fields=CONFIG['tool_fields']),
                          description=CONFIG['tool_description'])]
            llm = StreamingCallbackLLM(complete=partial(service.complete, task_id=CONFIG['task_id']),
                role='planner', max_tokens=CONFIG['max_tokens'], temperature=CONFIG['temperature'], trace=[])
            planner = planner_type(llm, '', '', tools, [])
            queue = asyncio.Queue()
            scheduler = TaskFetchingUnit()
            await asyncio.wait_for(asyncio.gather(
                planner.aplan({'input': CONFIG['question']}, queue, False),
                scheduler.aschedule(queue, None)), CONFIG['timeout_seconds'])
            await service.producer
            self.assertEqual(inputs, CONFIG['expected_tool_inputs'])
            self.assertFalse(dispatch[0])
            self.assertTrue(service.finished and service.released)
            self.assertEqual(llm.trace[0]['text'], plan)
            EVIDENCE['original_scheduler_tool_inputs'] = inputs
            EVIDENCE['planner_finished_at_tool_dispatch'] = dispatch
        asyncio.run(run())


if __name__ == '__main__':
    torch.manual_seed(CONFIG['seed'])
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(StreamingTests))
    Path(CONFIG['output']).write_text(json.dumps({'passed': result.wasSuccessful(), 'tests': result.testsRun,
        'scope': 'CPU protocol test with explicitly supplied token sequence, not model quality or GPU latency.',
        'evidence': EVIDENCE}, indent=2) + '\n')
    raise SystemExit(not result.wasSuccessful())
