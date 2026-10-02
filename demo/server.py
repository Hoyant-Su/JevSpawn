import argparse
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
from threading import Lock
import time
from urllib.parse import parse_qs, urlsplit
from uuid import uuid4


ROOT = Path(__file__).resolve().parents[1]


def metadata(sample):
    return {**{key: sample[key] for key in (
        'id', 'title', 'dataset', 'task_id', 'score', 'success', 'score_label')},
        'turns': len(sample['rounds'])}


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *args, settings, config_path, live_lock, **kwargs):
        self.settings, self.config_path, self.live_lock = settings, config_path, live_lock
        super().__init__(*args, directory=str(ROOT / settings['web']), **kwargs)

    def sample(self, identity):
        return json.loads((ROOT / self.settings['data'] / (identity + '.json')).read_text())

    def json_response(self, value):
        body = json.dumps(value, ensure_ascii=False).encode()
        self.send_response(200)
        self.send_header('Content-Type', 'application/json; charset=utf-8')
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def send_event(self, event):
        self.wfile.write(('data: ' + json.dumps(event, ensure_ascii=False) + '\n\n').encode())
        self.wfile.flush()

    def do_GET(self):
        request = urlsplit(self.path)
        if request.path == '/api/config':
            self.json_response({key: self.settings[key] for key in (
                'live_enabled', 'model_label', 'replay_interval_ms')})
        elif request.path == '/api/samples':
            self.json_response([metadata(self.sample(identity)) for identity in self.settings['samples']])
        elif request.path.startswith('/api/samples/'):
            identity = request.path.rsplit('/', 1)[-1]
            if identity not in self.settings['samples']:
                self.send_error(404)
                return
            sample = self.sample(identity)
            self.json_response({key: sample[key] for key in ('id', 'title', 'dataset', 'task_id',
                'context', 'score', 'success', 'score_label', 'description', 'instance_label')})
        elif request.path.startswith('/api/stream/'):
            identity = request.path.rsplit('/', 1)[-1]
            mode, = parse_qs(request.query)['mode']
            if identity not in self.settings['samples'] or mode not in ('replay', 'live'):
                self.send_error(400)
                return
            if mode == 'live' and not self.settings['live_enabled']:
                self.send_error(409, 'Start the server in live mode to enable GPU inference.')
                return
            self.send_response(200)
            self.send_header('Content-Type', 'text/event-stream; charset=utf-8')
            self.send_header('Cache-Control', 'no-cache')
            self.send_header('X-Accel-Buffering', 'no')
            self.end_headers()
            try:
                {'replay': self.replay, 'live': self.live}[mode](identity)
            except (BrokenPipeError, ConnectionResetError):
                pass
        elif request.path in ('/', '/index.html', '/style.css', '/app.js'):
            super().do_GET()
        else:
            self.send_error(404)

    def replay(self, identity):
        sample = self.sample(identity)
        self.send_event({'type': 'start', 'sample': {**metadata(sample), 'context': sample['context']},
                         'mode': 'replay'})
        for turn in sample['rounds']:
            time.sleep(self.settings['replay_interval_ms'] / 1000)
            self.send_event({'type': 'round', 'round': turn})
        self.send_event({'type': 'result', **{key: sample[key] for key in (
            'answer', 'score', 'success', 'selected_terminal')}})

    def live(self, identity):
        if not self.live_lock.acquire(blocking=False):
            self.send_event({'type': 'error', 'message': 'A live run is already using the model. Wait for completion.'})
            return
        try:
            self.run_live(identity)
        finally:
            self.live_lock.release()

    def run_live(self, identity):
        output = ROOT / self.settings['output'] / uuid4().hex
        output.mkdir(parents=True)
        events = output / (identity + '.jsonl')
        events.touch()
        with (output / 'worker.log').open('w') as log, events.open() as stream:
            process = subprocess.Popen([sys.executable, '-m', 'demo.launch',
                '--config', str(self.config_path), '--samples', identity, '--output', str(output)],
                cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            terminal = False
            try:
                while True:
                    line = stream.readline()
                    if line:
                        event = json.loads(line)
                        terminal |= event['type'] in ('result', 'error')
                        self.send_event(event)
                    elif process.poll() is not None:
                        break
                    else:
                        self.wfile.write(b': waiting for model\n\n')
                        self.wfile.flush()
                        time.sleep(self.settings['poll_interval_seconds'])
                if not terminal:
                    self.send_event({'type': 'error', 'message':
                        f'Inference worker exited with code {process.returncode}. See {output.relative_to(ROOT)}/worker.log.'})
            finally:
                if process.poll() is None:
                    os.killpg(process.pid, signal.SIGTERM)
                    process.wait()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--live', action='store_true')
    args = parser.parse_args()
    settings = json.loads(args.config.read_text())
    settings['live_enabled'] = args.live
    handler = partial(Handler, settings=settings, config_path=args.config.resolve(), live_lock=Lock())
    server = ThreadingHTTPServer((settings['host'], settings['port']), handler)
    print(f"JevSpawn demo: http://{settings['host']}:{settings['port']} ({'live and replay' if args.live else 'replay'})", flush=True)
    server.serve_forever()


if __name__ == '__main__':
    main()
