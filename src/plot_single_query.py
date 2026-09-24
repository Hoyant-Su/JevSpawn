import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / 'runs/decision-program-single-development-1024-001'
OUTPUT = ROOT / 'manuscript/figures'
METHODS = {
    'direct-tiled_shared': ('Direct shared', '#0072B2', 'o'),
    'direct-tiled_independent': ('Direct independent', '#0072B2', 's'),
    'tiled_shared': ('Generated shared', '#D55E00', 'o'),
    'tiled_independent': ('Generated independent', '#D55E00', 's'),
}


def main():
    analysis = json.loads((RUN / 'analysis.json').read_text())
    roots = analysis['roots']
    stage = json.loads((RUN / 'stage-decision.json').read_text())
    rows = []
    for method, (label, color, marker) in METHODS.items():
        values = [root['methods'][method] for root in roots]
        rows.append({
            'method': method, 'label': label, 'color': color, 'marker': marker,
            'seconds': np.mean([v['cold_program_inclusive_seconds'] for v in values]),
            'execution': np.mean([v['execution_seconds'] for v in values]),
            'planning': np.mean([v['compiler_seconds'] for v in values]),
            'quality': np.mean([v['ndcg_at_10'] for v in values]),
            'memory': [v['peak_allocated_bytes'] / 2**30 for v in values],
        })
    frontier = [r for r in rows if not any(
        q['seconds'] <= r['seconds'] and q['quality'] >= r['quality']
        and (q['seconds'] < r['seconds'] or q['quality'] > r['quality']) for q in rows)]
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 8,
                         'axes.spines.top': False, 'axes.spines.right': False,
                         'pdf.fonttype': 42, 'svg.fonttype': 'none'})
    fig = plt.figure(figsize=(7.2, 4.5), layout='constrained')
    grid = fig.add_gridspec(2, 2, width_ratios=[1.25, 1], hspace=0.15, wspace=0.12)
    ax = fig.add_subplot(grid[:, 0])
    cost = fig.add_subplot(grid[0, 1])
    memory = fig.add_subplot(grid[1, 1])
    ax.set_title('A   Quality and complete computation', loc='left', fontweight='bold', pad=12)
    ax.plot([r['seconds'] for r in frontier], [r['quality'] for r in frontier],
            '--', color='#555555', linewidth=1, label='Empirical frontier')
    offsets = [(7, -18), (7, 10), (-8, -27), (-6, 12)]
    for row, offset in zip(rows, offsets):
        ax.scatter(row['seconds'], row['quality'], color=row['color'], marker=row['marker'],
                   s=70, edgecolor='white', linewidth=0.8, zorder=3)
        ax.annotate(f"{row['label']}\n{row['seconds']:.2f} s, {row['quality']:.3f}",
                    (row['seconds'], row['quality']), xytext=offset, textcoords='offset points',
                    ha='left' if offset[0] > 0 else 'right', fontsize=7.5, color=row['color'])
    ax.set(xlabel='Mean query time including planning (s)', ylabel='Mean nDCG at 10',
           xlim=(14, 27), ylim=(0.11, 0.34))
    ax.grid(color='#e8e8e8', linewidth=0.6)
    ax.text(0.04, 0.97, 'Higher quality, less time', va='top', transform=ax.transAxes,
            fontsize=7.5, color='#555555')
    ax.legend(loc='lower left', frameon=False, fontsize=7)

    cost.set_title('B   Execution and planning', loc='left', fontweight='bold', pad=10)
    comparison = [rows[0], rows[2]]
    for index, row in enumerate(comparison):
        cost.barh(index, row['execution'], color=row['color'], height=0.45)
        cost.barh(index, row['planning'], left=row['execution'], color='#d8d8d8',
                  edgecolor='#777777', hatch='///', linewidth=0.5, height=0.45)
        cost.text(row['execution'] / 2, index, f"{row['execution']:.2f}",
                  color='white', ha='center', va='center', fontsize=8)
        if row['planning'] > 0:
            cost.text(row['execution'] + row['planning'] / 2, index,
                      f"{row['planning']:.2f}", ha='center', va='center', fontsize=7.5)
    cost.set(yticks=[0, 1], yticklabels=['Direct', 'Generated'], xlabel='Mean time (s)',
             xlim=(0, max(r['seconds'] for r in comparison) * 1.12), ylim=(-0.6, 1.6))
    cost.invert_yaxis()
    cost.legend(handles=[Line2D([], [], color='#777777', label='Hatched segment is planning')],
                frameon=False, loc='lower right', fontsize=7, handlelength=0)

    memory.set_title('C   Memory across queries', loc='left', fontweight='bold', pad=10)
    for index, row in enumerate(rows):
        values = np.asarray(row['memory'])
        memory.plot([values.min(), values.max()], [index, index], color=row['color'], alpha=0.4)
        memory.scatter(values, index + np.linspace(-0.1, 0.1, len(values)), s=12,
                       color=row['color'], marker=row['marker'], alpha=0.75)
        memory.text(values.max() + 0.4, index, f"{values.max():.2f}", va='center', fontsize=7)
    memory.set(yticks=range(len(rows)), yticklabels=[r['label'] for r in rows],
               xlabel='Peak allocated memory per query (GiB)', xlim=(19, 32))
    memory.tick_params(axis='y', labelsize=7)
    memory.invert_yaxis()
    memory.grid(axis='x', color='#e8e8e8', linewidth=0.6)
    fig.suptitle(f"One query, {stage['workers_per_root']:,} document workers, "
                 f"{stage['live_leaf_cap']} active at once\n"
                 f"{len(roots)} BRIGHT development queries with Qwen3.5-4B", fontsize=10)
    OUTPUT.mkdir(exist_ok=True)
    for extension in ['pdf', 'png', 'svg']:
        fig.savefig(OUTPUT / f'single_query_tradeoffs.{extension}', dpi=300)
    Image.open(OUTPUT / 'single_query_tradeoffs.png').convert('L').save(
        OUTPUT / 'single_query_tradeoffs_grayscale.png')
    (OUTPUT / 'single_query_tradeoffs.json').write_text(json.dumps({
        'source': str(RUN.relative_to(ROOT)), 'query_ids': [r['task_id'] for r in roots],
        'rows': rows, 'frontier': [r['method'] for r in frontier],
        'scope': 'Development means, not statistical confidence bounds. Planning plus execution excludes model startup. Memory points show all eight query peaks; labels show maxima.'}, indent=2) + '\n')


if __name__ == '__main__':
    main()
