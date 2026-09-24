"""Illustrate structured dependencies and the implemented finite execution schedule."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle


BLUE, GREEN, AMBER = '#E4F0F8', '#E2F2EC', '#FFF0D8'
EDGE = '#324653'


def box(ax, x, y, w, h, text, color='white', size=8):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.008',
                               facecolor=color, edgecolor=EDGE, linewidth=.8))
    ax.text(x+w/2, y+h/2, text, ha='center', va='center', fontsize=size)


def arrow(ax, start, end, *, dashed=False):
    ax.annotate('', xy=end, xytext=start,
                arrowprops={'arrowstyle': '-|>', 'lw': .85, 'color': EDGE,
                            'linestyle': '--' if dashed else '-'})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 8,
                         'pdf.fonttype': 42, 'mathtext.fontset': 'dejavusans'})
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 3.85),
                             gridspec_kw={'width_ratios': [1, 1.08]}, layout='constrained')
    for ax in axes:
        ax.set(xlim=(0, 1), ylim=(0, 1))
        ax.axis('off')
    ax = axes[0]
    ax.text(0, .99, 'A  Structured dependencies', weight='bold', va='top', fontsize=9)
    box(ax, .18, .81, .64, .11, 'Task $x$\nInterfaces and dependencies', BLUE, 7.5)
    for x in [.03, .60]:
        box(ax, x, .59, .37, .12, 'Finite worker\n$(q_j,\\mathcal{C}_j)$', GREEN)
    arrow(ax, (.39, .81), (.215, .72))
    arrow(ax, (.61, .81), (.785, .72))
    ax.text(.5, .745, 'independent decisions', ha='center', fontsize=7)
    box(ax, .02, .35, .40, .12, 'Conditional child\nfinite or generative', AMBER)
    box(ax, .59, .35, .38, .12, 'Dependent worker\nconsume $\\mu_j$', BLUE)
    arrow(ax, (.215, .59), (.215, .48))
    arrow(ax, (.785, .59), (.785, .48))
    ax.text(.215, .525, 'selected outcome', ha='center', fontsize=7,
            bbox={'facecolor': 'white', 'edgecolor': 'none', 'pad': 1}, zorder=10)
    ax.text(.785, .525, 'compact message', ha='center', fontsize=7,
            bbox={'facecolor': 'white', 'edgecolor': 'none', 'pad': 1}, zorder=10)
    box(ax, .26, .12, .48, .10, 'Aggregate task answer', BLUE)
    arrow(ax, (.22, .35), (.39, .23))
    arrow(ax, (.78, .35), (.61, .23))
    ax.text(.5, .035, 'Logical workers share model parameters $\\theta$',
            ha='center', fontsize=7.5)

    ax = axes[1]
    ax.text(0, .99, 'B  Shared finite computation', weight='bold', va='top', fontsize=9)
    ax.text(.01, .90, 'Exact common prefix', fontsize=7.5)
    ax.text(.68, .90, 'Suffixes', fontsize=7.5)
    for y, suffix in zip([.845, .795, .745], ['$s_1$', '$s_2$', '$s_F$']):
        ax.add_patch(Rectangle((.02, y), .57, .035, facecolor=BLUE, edgecolor=EDGE, lw=.6))
        ax.text(.30, y+.0175, '$p$', ha='center', va='center', fontsize=7)
        ax.add_patch(Rectangle((.61, y), .24, .035, facecolor=GREEN, edgecolor=EDGE, lw=.6))
        ax.text(.73, y+.0175, suffix, ha='center', va='center', fontsize=7)
    ax.text(.5, .665, 'One prefix evaluation per layer', ha='center', fontsize=7.5,
            bbox={'facecolor': 'white', 'edgecolor': 'none', 'pad': 1}, zorder=10)
    box(ax, .02, .50, .29, .11, 'Layer $\\ell$\nprefix state', BLUE)
    box(ax, .49, .50, .44, .11, 'Suffix tiles\nindependent mutable states', GREEN, 7.5)
    arrow(ax, (.31, .555), (.48, .555))
    ax.text(.395, .61, 'reuse', ha='center', fontsize=7)
    arrow(ax, (.305, .735), (.165, .62))
    arrow(ax, (.735, .735), (.71, .62))
    ax.plot([.04, .92], [.43, .43], color=EDGE, lw=.7, linestyle='--')
    ax.text(.48, .405, 'Release layer cache after its final use', ha='center', fontsize=7.5)
    arrow(ax, (.71, .495), (.71, .45))
    box(ax, .12, .245, .75, .095, 'Carry hidden states to layer $\\ell+1$', BLUE, 7.5)
    arrow(ax, (.71, .38), (.71, .35))
    box(ax, .12, .085, .75, .095, '$z_{jc}=W_{v(c)}h_j$     $\\mu_j=(\\hat{y}_j,p_j)$', AMBER, 8)
    arrow(ax, (.495, .24), (.495, .19))
    ax.text(.5, .025, 'Final layer     candidate rows only', ha='center', fontsize=7.5)
    args.output.mkdir(parents=True, exist_ok=True)
    for extension in ['pdf', 'png']:
        fig.savefig(args.output / ('execution_schedule.' + extension), dpi=300)


if __name__ == '__main__':
    main()
