from datetime import datetime, timezone
import json
from pathlib import Path
import urllib.request

from bs4 import BeautifulSoup

from project_paths import ROOT


OUTPUT = ROOT / 'results/reference_performance'
SOURCES = {
    'supergpqa': 'https://arxiv.org/html/2502.14739v1',
    'medxpertqa': 'https://arxiv.org/html/2501.18362v1',
    'processbench': 'https://arxiv.org/html/2412.06559v1',
    'zebralogic': 'https://arxiv.org/html/2502.01100v1',
    'multihoprag': 'https://arxiv.org/html/2401.15391v1',
    'ragtruth': 'https://arxiv.org/html/2401.00396v1',
    'bright': 'https://brightbenchmark.github.io/',
    'pubmedqa': 'https://pubmedqa.github.io/',
    'qwen35_4b': 'https://huggingface.co/Qwen/Qwen3.5-4B',
    'qwen25_7b': 'https://huggingface.co/Qwen/Qwen2.5-7B-Instruct',
    'qwen25_32b': 'https://huggingface.co/Qwen/Qwen2.5-32B-Instruct',
    'squad2': 'https://arxiv.org/html/1806.03822v1',
    'race': 'https://arxiv.org/html/1704.04683v2',
    'scifact': 'https://arxiv.org/html/2004.14974v1',
    'humaneval': 'https://arxiv.org/html/2107.03374v2',
    'mbpp': 'https://arxiv.org/html/2108.07732v1',
    'aqua': 'https://arxiv.org/html/1705.04146v1',
    'qwen25_report': 'https://arxiv.org/html/2412.15115v1',
    'cot': 'https://arxiv.org/html/2201.11903v6',
    'bert': 'https://arxiv.org/html/1810.04805v2',
    'bright_paper': 'https://arxiv.org/html/2407.12883v1',
}


def fetch(item):
    name, url = item
    request = urllib.request.Request(url, headers={'User-Agent': 'JevSpawn research reference lookup'})
    with urllib.request.urlopen(request, timeout=40) as response:
        html = response.read().decode()
    (OUTPUT / 'sources' / (name + '.html')).write_text(html)
    soup = BeautifulSoup(html, 'html.parser')
    tables = []
    for index, table in enumerate(soup.find_all('table')):
        data = [[cell.get_text(' ', strip=True) for cell in row.find_all(['th', 'td'], recursive=False)]
                for row in table.find_all('tr')]
        caption = table.find_previous('figcaption')
        tables.append(dict(index=index, rows=data, caption=caption.get_text(' ', strip=True) if caption else None))
    record = dict(url=url, retrieved_utc=datetime.now(timezone.utc).isoformat(), tables=tables)
    (OUTPUT / 'sources' / (name + '.json')).write_text(json.dumps(record, indent=2) + '\n')
    print(name, len(tables), flush=True)
    return name


if __name__ == '__main__':
    # arXiv HTML requests are sequential; no API fan-out against rate-limited hosts.
    for item in SOURCES.items():
        fetch(item)
