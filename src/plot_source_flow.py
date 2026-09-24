import argparse
from collections import defaultdict
import json
from pathlib import Path
import textwrap

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import PathPatch
from matplotlib.path import Path as Curve

from methods.source_interfaces.analysis import hierarchy_statistics
from methods.source_interfaces.schema import NONE


COLORS = ['#0072B2', '#D55E00', '#009E73', '#CC79A7']


def load_case(run, task_id):
    report = json.loads((run / 'evaluation.json').read_text())
    rows = report['methods']['streamed']['queries']
    index = [row['task_id'] for row in rows].index(task_id)
    outcome = json.loads((run / 'streamed/measured' / f'{index:03d}' / 'outcome.json').read_text())
    result = outcome['primary']
    calls = json.loads((Path(outcome['attempt']) / 'calls.json').read_text())
    topology = hierarchy_statistics(result, calls)
    nodes, edges, owners = [], [], {}
    for call in calls:
        if call['kind'] != 'worker':
            continue
        level = call['context']['level']
        for fields, values in zip(call['context']['groups'], call['result']['groups']):
            for field, value in zip(fields, values):
                node = {'index': len(nodes), 'field': field['id'], 'level': level,
                        'choice': value['choice'], 'candidates': [o['id'] for o in field['options'] if o['id'] != NONE]}
                if level:
                    edges.extend((owners[(field['id'], source)], node['index']) for source in node['candidates'])
                nodes.append(node)
                if value['choice'] != NONE:
                    owners[(field['id'], value['choice'])] = node['index']
    assert len(nodes) == result['worker_invocations']
    terminal = {r['id']: owners[(r['id'], r['source_id'])] for r in result['references'] if r['source_id'] is not None}
    return report, outcome, topology, nodes, edges, terminal


def curve(ax, start, end, color, width, alpha):
    x0, y0 = start
    x1, y1 = end
    middle = (x0 + x1) / 2
    path = Curve([(x0, y0), (middle, y0), (middle, y1), (x1, y1)],
                 [Curve.MOVETO, Curve.CURVE4, Curve.CURVE4, Curve.CURVE4])
    ax.add_patch(PathPatch(path, fill=False, color=color, lw=width, alpha=alpha, zorder=1))


def draw(report, outcome, topology, nodes, edges, terminal, palette):
    result = outcome['primary']
    references = result['references']
    fields = [r['id'] for r in references]
    fig = plt.figure(figsize=(7.2, 7.4), layout='constrained')
    grid = fig.add_gridspec(2, 1, height_ratios=[1.65, 1])
    ax = fig.add_subplot(grid[0])
    ax.set(xlim=(0, 100), ylim=(-.55, len(fields) + .75))
    ax.axis('off')
    ax.set_title('A  Evidence flow for one source-dependent answer', loc='left', fontweight='bold', pad=18)
    ax.text(0, len(fields) + .57,
            f'{topology["source_units"]} source units, {len(fields)} generated questions, '
            f'{len(nodes)} finite workers, one model replica', fontsize=8)
    positions = {}
    grouped = defaultdict(list)
    for node in nodes:
        grouped[(node['field'], node['level'])].append(node)
    stage_x = [38, 59, 72]
    for lane, reference in enumerate(references):
        center = len(fields) - lane - .5
        color = palette[lane]
        ax.text(0, center + .36, '\n'.join(textwrap.wrap(reference['question'], 39)),
                fontsize=7.2, va='top', linespacing=1.25)
        ax.text(0, center + .48, f'Q{lane + 1}', color=color, fontweight='bold', fontsize=8)
        for level in range(len(result['levels'])):
            group = grouped[(reference['id'], level)]
            for offset, node in enumerate(group):
                if level == 0:
                    x = 32 + 1.65 * (offset % 8)
                    y = center + .28 - .105 * (offset // 8)
                else:
                    x = stage_x[level]
                    y = center + (.23 - .46 * offset / (len(group) - 1) if len(group) > 1 else 0)
                positions[node['index']] = (x, y)
            ax.text(stage_x[level], center - .41, f'{len(group)} worker' + ('s' if len(group) != 1 else '') if group else 'No further selection',
                    ha='center', fontsize=6.8, color='#4d4d4d')
        if reference['source_id'] is not None:
            point = terminal[reference['id']]
            curve(ax, positions[point], (84, center), color, 1.15, 1)
            ax.text(85, center, reference['source_id'], va='center', fontsize=7,
                    bbox={'boxstyle': 'round,pad=.35', 'facecolor': 'white', 'edgecolor': color, 'linewidth': 1})
    for source, target in edges:
        a, b = nodes[source], nodes[target]
        lane = fields.index(a['field'])
        selected = a['choice'] == references[lane]['source_id']
        curve(ax, positions[source], positions[target], palette[lane] if selected else '#aaaaaa',
              1.05 if selected else .45, 1 if selected else .45)
    for node in nodes:
        x, y = positions[node['index']]
        color = palette[fields.index(node['field'])]
        none = node['choice'] == NONE
        ax.scatter(x, y, s=14 if node['level'] == 0 else 27,
                   marker='x' if none else 'o', color='#ababab' if none else color,
                   linewidths=.55, zorder=3)
    for x, label in zip(stage_x, ['Passage groups', 'Child\nreferences', 'Final\nselection']):
        ax.text(x, len(fields) + .2, label, ha='center', fontsize=7.5, fontweight='bold')
    ax.text(91, len(fields) + .2, 'Source returned', ha='center', fontsize=7.5, fontweight='bold')
    ax.legend(handles=[Line2D([], [], marker='o', color='#555555', linestyle='', markersize=4, label='Returns a source'),
                       Line2D([], [], marker='x', color='#999999', linestyle='', markersize=4, label='Returns no evidence'),
                       Line2D([], [], color='#555555', lw=1.2, label='Selected source path')],
              loc='lower left', bbox_to_anchor=(0, -.04), ncol=3, frameon=False, fontsize=7.2,
              handlelength=1.5, columnspacing=1.4)
    ax.text(0, -.29, f'Answer after reading selected text  {result["answer"]}', fontsize=8, fontweight='bold')
    ax.text(0, -.47, f'Answer without selected text  {outcome["interventions"]["no_references"]["answer"]}', fontsize=8)

    bx = fig.add_subplot(grid[1])
    rows = report['methods']['streamed']['queries']
    max_depth = max(row['levels'] for row in rows)
    bx.set(xlim=(-.6, max_depth + 3.4), ylim=(len(rows) - .5, -1.7))
    bx.axis('off')
    bx.set_title('B  Expansion and message utility across all development questions', loc='left', fontweight='bold', pad=12)
    headings = ['Task'] + [f'Level {i + 1}' for i in range(max_depth)] + ['Total', 'Answer', 'Without sources']
    for col, label in enumerate(headings):
        bx.text(col - .1, -1.05, label, ha='center', fontsize=7.5, fontweight='bold')
    maximum = max(row['workers'] for row in rows)
    for y, row in enumerate(rows):
        bx.axhspan(y - .46, y + .46, color='#f2f2f2' if y % 2 == 0 else 'white', zorder=0)
        bx.text(-.1, y, row['task_id'].split('/')[-1], va='center', ha='center', fontsize=8)
        values = row['topology']['workers_per_level']
        for level in range(max_depth):
            x = level + .9
            if level < len(values):
                count = values[level]
                bx.scatter(x, y, s=45 + 410 * count / maximum, facecolor=palette[0], alpha=.24, edgecolor='none')
                bx.text(x, y, str(count), ha='center', va='center', fontsize=8)
            else:
                bx.text(x, y, '–', color='#888888', ha='center', va='center', fontsize=8)
        bx.text(max_depth + .9, y, str(row['workers']), ha='center', va='center', fontsize=8)
        for offset, name in enumerate(['primary', 'no_references']):
            correct = row['scores'][name]['exact_match']
            bx.text(max_depth + 1.9 + offset, y, 'Correct' if correct else 'Incorrect',
                    color=palette[2] if correct else '#666666', ha='center', va='center', fontsize=7.5)
    return fig


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--task-id', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 8, 'axes.titlesize': 9,
                         'pdf.fonttype': 42, 'svg.fonttype': 'none'})
    report, outcome, topology, nodes, edges, terminal = load_case(args.run, args.task_id)
    fig = draw(report, outcome, topology, nodes, edges, terminal, COLORS)
    for suffix in ['.pdf', '.png', '.svg']:
        fig.savefig(args.output.with_suffix(suffix), dpi=300)
    plt.close(fig)
    gray = draw(report, outcome, topology, nodes, edges, terminal, ['#333333', '#555555', '#777777', '#999999'])
    gray.savefig(args.output.with_name(args.output.name + '_grayscale').with_suffix('.png'), dpi=180)
    plt.close(gray)
    args.output.with_suffix('.json').write_text(json.dumps({
        'run': str(args.run), 'task_id': args.task_id, 'topology': topology, 'nodes': nodes,
        'edges': edges, 'terminal': terminal,
        'selection': 'An illustrative source-dependent correct answer. Panel B includes every development question. Node and edge identities come from actual finite model calls. No synthetic workers or timing estimates.'}, indent=2) + '\n')
    print({'task_id': args.task_id, 'nodes': len(nodes), 'edges': len(edges), 'output': str(args.output)})


if __name__ == '__main__':
    main()
