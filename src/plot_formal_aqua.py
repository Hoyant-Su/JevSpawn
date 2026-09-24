import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
STYLES = {'ReAct': ('#0072B2', 'o'), 'LLMCompiler': ('#E69F00', 's'),
          'LatentMAS': ('#009E73', 'D'), 'AgentPrune': ('#CC79A7', '^'),
          'DyFlow': ('#D55E00', 'v')}


def main():
    rows = json.loads((ROOT / 'manuscript/tables/published_aqua.json').read_text())['records']
    uncertainty = json.loads((ROOT / 'results/paired_formal_aqua.json').read_text())
    frontier = sorted([r for r in rows if not any(
        q['elapsed_seconds'] <= r['elapsed_seconds'] and q['accuracy'] >= r['accuracy']
        and (q['elapsed_seconds'] < r['elapsed_seconds'] or q['accuracy'] > r['accuracy'])
        for q in rows)], key=lambda r: r['elapsed_seconds'])
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 8, 'axes.spines.top': False,
                         'axes.spines.right': False, 'pdf.fonttype': 42, 'svg.fonttype': 'none'})
    fig = plt.figure(figsize=(7.2, 5.8))
    grid = fig.add_gridspec(2, 2, height_ratios=[1.6, 1], left=0.1, right=0.975,
                           top=0.86, bottom=0.15, wspace=0.36, hspace=0.54)
    quality = fig.add_subplot(grid[0, 0])
    differences = fig.add_subplot(grid[0, 1])
    costs = fig.add_subplot(grid[1, :])
    quality.plot([r['elapsed_seconds'] / 60 for r in frontier],
                 [100 * r['accuracy'] for r in frontier], '--', color='#777777', linewidth=1)
    for row in rows:
        name = row['method']
        color, marker = STYLES[name]
        y = row['accuracy'] * 100
        lo, hi = uncertainty['methods'][name]['interval_percentage']
        quality.errorbar(row['elapsed_seconds'] / 60, y, yerr=[[y-lo], [hi-y]], fmt=marker,
                         color=color, markersize=6, capsize=3, linewidth=1)
    intervals = [uncertainty['methods'][r['method']]['interval_percentage'] for r in rows]
    quality.set(xlabel='Complete workload time (min)', ylabel='Accuracy (%)',
                xlim=(0, max(r['elapsed_seconds'] / 60 for r in rows) * 1.12),
                ylim=(min(lo for lo, hi in intervals) - 5, max(hi for lo, hi in intervals) + 5))
    quality.set_title('A   Quality and time', loc='left', fontweight='bold')
    quality.grid(color='#e8e8e8', linewidth=0.6)
    quality.text(0.04, 0.95, 'Dashed line is the empirical frontier', transform=quality.transAxes,
                 va='top', fontsize=6.5, color='#555555')
    pairs = [row for row in uncertainty['pairs'] if row['left'] == 'ReAct']
    for index, pair in enumerate(pairs):
        x = pair['difference_percentage_points']
        lo, hi = pair['paired_interval_percentage_points']
        color, marker = STYLES[pair['right']]
        differences.errorbar(x, index, xerr=[[x-lo], [hi-x]], fmt=marker,
                             color=color, markersize=6, capsize=3, linewidth=1)
        differences.annotate(f'{x:+.2f} pp', (x, index), xytext=(0, 8),
                             textcoords='offset points', ha='center', fontsize=7)
    differences.axvline(0, color='#777777', linewidth=0.8, linestyle=':')
    limits = [p['paired_interval_percentage_points'] for p in pairs]
    differences.set(yticks=range(len(pairs)), yticklabels=[p['right'] for p in pairs],
                    xlabel='ReAct minus comparator (pp)',
                    xlim=(min(0, min(lo for lo, hi in limits)) - 2,
                          max(0, max(hi for lo, hi in limits)) + 3),
                    ylim=(-0.65, len(pairs) - .35))
    differences.invert_yaxis()
    differences.set_title('B   Paired accuracy differences', loc='left', fontweight='bold')
    differences.tick_params(axis='y', labelsize=7)
    differences.grid(axis='x', color='#e8e8e8', linewidth=0.6)

    costs.axis('off')
    costs.set_title('C   Realized outputs and memory', loc='left', fontweight='bold', pad=9)
    cells = [[r['method'], f"{r['correct']}/{r['tasks']}", f"{r['valid']}/{r['tasks']}",
              f"{r['output_tokens']:,}", f"{r['peak_allocated_bytes'] / 2**30:.2f}",
              f"{r['max_itl_ms']:.2f}"] for r in rows]
    table = costs.table(cellText=cells,
                        colLabels=['Method', 'Correct', 'Valid', 'Output tokens', 'Peak GiB', 'Max ITL, ms'],
                        colWidths=[.21, .13, .13, .2, .15, .18], cellLoc='right', loc='upper center')
    table.auto_set_font_size(False)
    table.set_fontsize(7.5)
    table.scale(1, 1.25)
    for (row, column), cell in table.get_celld().items():
        cell.set_linewidth(0)
        if row == 0:
            cell.set_facecolor('#edf0f3')
            cell.set_text_props(weight='bold')
        if column == 0:
            cell.set_text_props(ha='left')
            if row:
                cell.get_text().set_color(STYLES[rows[row-1]['method']][0])
    handles = [Line2D([], [], color=STYLES[r['method']][0], marker=STYLES[r['method']][1],
                      linestyle='none', label=r['method']) for r in rows]
    fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(.52, .935),
               ncol=len(rows), frameon=False)
    fig.suptitle(f"Published agent operating points on {uncertainty['tasks']} AQuA questions", fontsize=11, y=.99)
    fig.text(.1, .035,
             'Qwen3.5-4B, BF16, request batch cap 8. Intrinsic method budgets differ.\n'
             'Bars show unadjusted 95% question bootstrap intervals. Timing has one realization.\n'
             'AgentPrune topology learning and model startup are excluded from workload time.',
             fontsize=7, color='#555555', linespacing=1.5)
    output = ROOT / 'manuscript/figures'
    for extension in ['pdf', 'png', 'svg']:
        fig.savefig(output / f'published_aqua_tradeoffs.{extension}', dpi=300)
    Image.open(output / 'published_aqua_tradeoffs.png').convert('L').save(
        output / 'published_aqua_tradeoffs_grayscale.png')
    (output / 'published_aqua_tradeoffs.json').write_text(json.dumps({
        'sources': ['manuscript/tables/published_aqua.json', 'results/paired_formal_aqua.json'],
        'empirical_frontier': [r['method'] for r in frontier],
        'interpretation': 'Observed method operating points under distinct intrinsic budgets. No equal-compute superiority or optimized serving claim.'}, indent=2) + '\n')


if __name__ == '__main__':
    main()
