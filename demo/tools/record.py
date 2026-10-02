import argparse
import asyncio
import json
from pathlib import Path
import struct
import time

from playwright.async_api import async_playwright


ROOT = Path(__file__).resolve().parents[2]


async def capture(page, settings, config, stop):
    encoder = await asyncio.create_subprocess_exec(settings['encoder_python'], '-m',
        'demo.tools.encode_video', '--config', str(config), stdin=asyncio.subprocess.PIPE)
    started = time.monotonic()
    while not stop.is_set():
        png = await page.screenshot()
        elapsed = time.monotonic() - started
        encoder.stdin.write(struct.pack('!Id', len(png), elapsed) + png)
        await encoder.stdin.drain()
        await asyncio.sleep(1 / settings['frame_rate'])
    encoder.stdin.close()
    assert await encoder.wait() == 0, 'Video encoder failed.'


async def record(settings, config):
    output = ROOT / settings['screenshots']
    output.mkdir(parents=True, exist_ok=True)
    video_path = ROOT / settings['video']
    video_path.parent.mkdir(parents=True, exist_ok=True)
    async with async_playwright() as playwright:
        browser = await playwright.chromium.launch()
        context = await browser.new_context(viewport=settings['viewport'])
        page = await context.new_page()
        page.set_default_timeout(settings['timeout_ms'])
        errors, results = [], []
        page.on('pageerror', lambda error: errors.append(str(error)))
        await page.goto(settings['url'])
        await page.wait_for_function("window.demo && window.demo.status === 'ready'")
        stop = asyncio.Event()
        recording = asyncio.create_task(capture(page, settings, config, stop))
        sample_ids = await page.locator('.sample').evaluate_all('(nodes) => nodes.map(node => node.dataset.id)')
        for identity in sample_ids:
            await page.locator(f'.sample[data-id="{identity}"]').click()
            await page.wait_for_function("window.demo.status === 'ready'")
            await page.wait_for_timeout(settings['context_pause_ms'])
            await page.select_option('#mode', settings['mode'])
            await page.locator('#run').click()
            await page.wait_for_function("window.demo.status === 'complete' || window.demo.status === 'error'")
            status = await page.evaluate('window.demo.status')
            assert status == 'complete', await page.locator('#error').inner_text()
            sample = await page.evaluate('window.demo.sample')
            rounds = await page.evaluate('window.demo.rounds')
            assert rounds == sample['turns'], (identity, rounds, sample['turns'])
            await page.screenshot(path=str(output / (identity + '.png')))
            results.append({'sample': identity, 'rounds': rounds,
                            'result': await page.locator('#result').inner_text()})
            print(identity, rounds, 'rounds displayed', flush=True)
            await page.wait_for_timeout(settings['result_pause_ms'])
        stop.set()
        await recording
        await context.close()
        await browser.close()
    (ROOT / settings['report']).write_text(json.dumps({'browser_errors': errors, 'samples': results}, indent=2) + '\n')
    assert not errors, errors
    print(video_path, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    args = parser.parse_args()
    asyncio.run(record(json.loads(args.config.read_text()), args.config.resolve()))
