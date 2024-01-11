#!/usr/bin/env python
import argparse
import pathlib
import json

import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def cmd_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', '-s', type=int, default=0)
    parser.add_argument('--noise_std', '-n', type=float, default=.02)
    parser.add_argument('--nb_subj', type=int, default=2)
    parser.add_argument('--lag', '-l', type=int, default=1)
    parser.add_argument('--graph', '-g', type=pathlib.Path, required=True, help='path to graph (json)')
    parser.add_argument('--timepoints', '-t', type=int, default=6)
    parser.add_argument('--kind', '-k', required=True, choices=['sigmoid', 'brownian'])
    parser.add_argument('--gap', type=int, default=1, help='gap between timepoints')
    parser.add_argument('--outdir', '-o', type=pathlib.Path, required=True)
    return parser.parse_args()

# a custom implementation of the sigmoid function
# Returns a transformed value based on the sigmoid function
def sigmoid(x, x0=0, L=1, k=1, offset=0):
    return (L/2)*(1 + np.tanh(k*(x-x0)/2)) + offset


def generate_subj(graph, rng, noise_std, kind_):
    t = np.arange(200) - 50  # pad front and back by 50, for time-shifting
    series = {}
    for n in nx.topological_sort(graph):
        parents = list(graph.predecessors(n))

        if not parents:
            if kind_ == 'sigmoid':
                y = sigmoid(t, x0=rng.uniform(40, 60), L=rng.uniform(1, 2), k=rng.uniform(.1, .3))
            elif kind_ == 'brownian':
                y = np.cumsum(rng.normal(loc=0, scale=1, size=len(t)))
        else:
            edges = [graph[p][n] for p in parents]
            y = sum(e['sign']*e['mag']*np.roll(series[p], shift=e['lag']) for p, e in zip(parents, edges))
        y += rng.uniform(-.5, .5)  # bias

        series[n] = y + rng.normal(scale=noise_std, size=y.shape)  # additive noise

    return t[50:], {k: v[50:] for k, v in series.items()}


def plot_graph(graph, axe):
    # pos = nx.circular_layout(graph)
    pos = nx.planar_layout(graph)
    # edge_labels = {e: f'{graph.edges[e]["sign"]*graph.edges[e]["mag"]:.1f}' for e in graph.edges()}
    edge_labels = {}
    # nx.draw(graph, pos, with_labels=True, node_size=1000, arrowsize=15, font_weight='bold', ax=axe)
    nx.draw(graph, pos, with_labels=True, ax=axe)
    nx.draw_networkx_edge_labels(graph, pos, edge_labels=edge_labels, font_color='red')


def skel_attr(skeleton, rng, lag):
    for s, t in skeleton.edges:
        skeleton[s][t]['mag'] = rng.uniform(.5, 1)
        skeleton[s][t]['lag'] = lag
        if 'sign' not in skeleton[s][t]:
            skeleton[s][t]['sign'] = 1 if rng.binomial(1, .5) else -1

    return skeleton


def sample(x, series, t_range):
    out = {label: s[t_range] for label, s in series.items()}
    out['AGE'] = x[t_range]
    out['Year'] = out['AGE'] - out['AGE'].min()
    return pd.DataFrame(out)


def main(args, rng):
    with open(args.graph) as fh:
        dictj = json.load(fh)
        dictj = {k[-1]: [i[-1] for i in v] for k, v in dictj.items()}
        skel = nx.DiGraph(dictj)

    G = skel_attr(skel, rng, args.lag)
    fig, axes = plt.subplots(1, 3, figsize=(9, 3))
    plot_graph(G, axes[-1])

    data = []
    dt = np.arange(args.timepoints, dtype=int)*args.gap
    for idx in range(args.nb_subj):
        x, series = generate_subj(G, rng, args.noise_std, 'brownian' if (idx + 1) % 2 else 'sigmoid')
        t_range = rng.integers(30, 70) + dt
        if idx < len(axes) - 1:
            for node, y in series.items():
                axes[idx].plot(x[20:100], y[20:100], label=node)
            axes[idx].legend()
            for pos in t_range:
                axes[idx].axvline(pos, color='b', alpha=.5)

        sframe = sample(x, series, t_range)
        sframe['RID'] = idx
        data.append(sframe)

    columns = sorted(list(data[0].columns))
    data = pd.concat([sf[columns] for sf in data])
    print(data.head(n=10))

    fig.tight_layout()
    return fig, data


if __name__ == '__main__':
    args = cmd_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    fig, data = main(args, rng=np.random.default_rng(args.seed))
    stem = f'{args.graph.stem}_nstd{args.noise_std}_lag{args.lag}_gap{args.gap}_tp{args.timepoints:02d}_seed{args.seed}'
    plt.show()
    #assert False
    data.columns = [f'Feat{col}' if col.isdigit() else col for col in data.columns]
    data.to_csv(args.outdir/f'{stem}.csv', index=False)
    fig.savefig(args.outdir/f'{stem}.png')

    config = {'csv': f'genData/{stem}.csv'}
    with open(args.graph) as fh:
        adj_list = json.load(fh)
        config.update({'feats': list(adj_list.keys()), 'groundtruth': adj_list})
    with open(args.outdir/f'{stem}.json', 'w') as fh:
        json.dump(config, fh, indent=4)
