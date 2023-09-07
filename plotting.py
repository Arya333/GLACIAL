import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import pydot


def agg_trace(traces, val_traces=None):
    def aux(array):
        max_len = max(a.shape[1] for a in array)
        y_ = np.vstack([np.pad(a, [(0,0), (0, max_len-a.shape[1])], constant_values=np.nan) for a in array])
        y_avg = np.nanmean(y_, axis=0)
        y_err = np.nanstd(y_, axis=0)
        x = np.arange(y_.shape[-1])
        return x, y_avg, y_avg - y_err, y_avg + y_err

    fig, ax = plt.subplots(1, 1)
    t_x, t_est, t_lo, t_hi = aux(traces)
    ax.fill_between(t_x, t_lo, t_hi, color='b', alpha=0.5)
    ax.plot(t_x, t_est, color='b', label='train')
    if val_traces:
        v_x, v_est, v_lo, v_hi = aux(val_traces)
        ax.fill_between(v_x, v_lo, v_hi, color='g', alpha=0.5)
        ax.plot(v_x, v_est, color='g', label='valid')
    ax.legend()
    return fig


def xplot_binom(st, features, th):
    D = len(features)
    adj_mat = np.zeros((D, D), dtype=int)
    for i in range(D):
        for j in range(D):
            if i == j:
                continue
            x = st[:, i, j]
            result = stats.binomtest((x<0).sum(), len(x), alternative='greater')  # prob > .5
            print(result)
            adj_mat[i, j] = result.pvalue < th/D**2

    return adj_mat


def xplot(st, pv, features, savepath, th):
    D = len(features)
    adj_mat = np.zeros((D, D), dtype=int)
    for i, feat1 in enumerate(features):
        for j, feat2 in enumerate(features):
            if i == j:
                continue

            fig, axes = plt.subplots(1, 3, figsize=(9, 3))
            x, y = st[:, i, j], pv[:, i, j]
            axes[0].hist(x, bins=20, rwidth=.8)
            st0, pv0 = stats.ttest_1samp(x, popmean=0)
            adj_mat[i, j] = (st0 < 0) & (pv0 < th/D**2)
            axes[0].set_title(f'st={st0:.2f} pv={pv0:.2e}')
            axes[1].hist(y, bins=20, rwidth=.8)
            _, _, _, im = axes[2].hist2d(x, y, bins=[np.arange(-10, 11), np.arange(-20, 1)])
            axes[2].axvline(x=0, color='white')
            fig.colorbar(im)
            fig.suptitle(f'{feat1} -> {feat2}')
            fig.tight_layout()
            fig.savefig(savepath/f'{feat1}-{feat2}.png')
            plt.close(fig)

    return adj_mat


def edge_color(u, v, direction, true_graph=None):
    if true_graph is None:
        return 'black'
    if direction in {'---', 'o-o', '<->'}:
        if true_graph.has_edge(u, v) or true_graph.has_edge(v, u):
            return 'magenta'
    if direction in {'-->', 'o->'}:
        if true_graph.has_edge(v, u):
            return 'red'  # inverted edge
        if true_graph.has_edge(u, v):
            return 'blue'  # correct direct edge
    elif direction in {'<--', '<-o'}:
        if true_graph.has_edge(u, v):
            return 'red'  # inverted edge
        if true_graph.has_edge(v, u):
            return 'blue'  # correct direct edge
    return 'black'


def cgraph(digraph, out_fname, true_graph=None):
    graph = pydot.Dot('my_graph', graph_type='graph', bgcolor='white')
    NODE_TBL = {
        'PTEDUCAT': ['EDU', 'black'],
        'PTGENDER': ['SEX', 'black'],
        'AGE': ['AGE', 'black'],

        'APGEN1': ['APGEN1', 'red'],
        'APGEN2': ['APGEN2', 'red'],
        'APOE': ['APOE', 'red'],

        'FDG': ['FDG', 'blue'],
        'PTAU': ['PTAU', 'blue'],
        'ABETA': ['ABETA', 'blue'],

        'DX': ['DX', 'green'],

        'ADAS13': ['ADAS13', 'black'],
        'MMSE': ['MMSE', 'black'],
        'MOCA': ['MOCA', 'black'],

        'Ventricles': ['Ventricles', 'tomato'],
        'Hippocampus': ['Hippocampus', 'tomato'],
        'WholeBrain': ['WholeBrain', 'tomato'],
        'Entorhinal': ['Entorhinal', 'tomato'],
        'Fusiform': ['Fusiform', 'tomato'],
        'MidTemp': ['MidTemp', 'tomato'],
        'ICV': ['ICV', 'tomato'],
    }

    for node in digraph.nodes:
        label, color = NODE_TBL.get(node, [node, 'black'])
        graph.add_node(pydot.Node(label, shape='box', color=color, penwidth=2))

    for src, dst in digraph.edges:
        frequency = digraph.edges[src, dst]['frequency']
        attrs = {'label': '' if frequency is None else f'{frequency:.2}'}

        direction = digraph.edges[src, dst]['direction']
        attrs['color'] = edge_color(src, dst, direction, true_graph)
        if direction == '---':
            pass
        elif direction == '-->':
            attrs.update({'dir': 'both', 'arrowtail': 'none', 'arrowhead': 'normal'})
        elif direction == '<--':
            attrs.update({'dir': 'both', 'arrowtail': 'normal', 'arrowhead': 'none'})
        elif direction == '<->':
            attrs.update({'dir': 'both', 'arrowtail': 'normal', 'arrowhead': 'normal'})
        elif direction == '<-o':
            attrs.update({'dir': 'both', 'arrowtail': 'normal', 'arrowhead': 'dot'})
        elif direction == 'o->':
            attrs.update({'dir': 'both', 'arrowtail': 'dot', 'arrowhead': 'normal'})
        elif direction == 'o-o':
            attrs.update({'dir': 'both', 'arrowtail': 'dot', 'arrowhead': 'dot'})
        else:
            raise ValueError(direction)

        src, dst = NODE_TBL.get(src, [src, ''])[0], NODE_TBL.get(dst, [dst, ''])[0]
        graph.add_edge(pydot.Edge(src, dst, **attrs))

    # graph.write_pdf(out_fname)
    graph.write_png(out_fname, prog=['dot','-Gdpi=100!'])
