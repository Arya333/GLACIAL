import numpy as np
import pandas as pd
import scipy.stats as stats
import networkx as nx

import metrics
import plotting


def check_grandpa(direction, eff_size):
    out_dir = direction.copy()
    G = nx.DiGraph(direction)
    for u, v in G.edges:
        for p in nx.all_simple_paths(G, u, v, 5):  # for tractability
            if len(p) == 2:
                continue
            if eff_size[u, v] < min(eff_size[edge] for edge in zip(p, p[1:])):
                out_dir[u, v] = 0
                break

        for p in nx.all_simple_paths(G, v, u, 5):  # for tractability
            if len(p) == 2:
                continue
            if eff_size[u, v] < min(eff_size[edge] for edge in zip(p, p[1:])):
                out_dir[u, v] = 0
                break

    return out_dir


def frame2cgraph(frame, out_fname, true_graph=None):
    node_list = frame.columns
    graph = nx.DiGraph()
    graph.add_nodes_from(node_list)

    prob = frame.to_numpy()
    influence = prob > 0

    for i, v1 in enumerate(node_list):
        for j, v2 in zip(range(i+1, len(node_list)), node_list[i+1:]):
            if influence[i][j] and not influence[j][i]:
                graph.add_edge(v1, v2, direction='-->', frequency=prob[i,j])
            elif not influence[i][j] and influence[j][i]:
                graph.add_edge(v1, v2, direction='<--', frequency=prob[j,i])
            elif influence[i][j] and influence[j][i]:
                graph.add_edge(v1, v2, direction='---', frequency=min(prob[i,j], prob[j,i]))

    plotting.cgraph(graph, out_fname, true_graph)
    return graph


def causal_graph(aux_mse, main_mse, features, outpath, true_graph=None):
    D = aux_mse.shape[-1]
    d_mse = np.where(np.isnan(aux_mse), 0, aux_mse - main_mse)
    percentage = (d_mse > 0).mean(axis=0)
    st1, pv1 = stats.ttest_1samp(d_mse, popmean=0, axis=0)
    pos1 = (st1 > 0) & (pv1 < 5e-2/D**2)

    arrows = np.where(pos1, percentage, 0)
    s1_graph = frame2cgraph(pd.DataFrame(data=arrows, columns=features), outpath/f'v1_s1.png', true_graph)

    pos2 = np.where(pos1 & pos1.T, st1 > st1.T, pos1)
    arrows = np.where(pos2, percentage, 0)
    s2_graph = frame2cgraph(pd.DataFrame(data=arrows, columns=features), outpath/f'v1_s2.png', true_graph)

    arrows = np.where(check_grandpa(pos2, st1), percentage, 0)
    s3_graph = frame2cgraph(pd.DataFrame(data=arrows, columns=features), outpath/f'v1_s3.png', true_graph)

    if true_graph is not None:
        frame = pd.DataFrame([
            metrics.score(s1_graph, true_graph, 'S1'),
            metrics.score(s2_graph, true_graph, 'S2'),
            metrics.score(s3_graph, true_graph, 'S3'),
        ])
        frame.to_csv(outpath/'score.csv', index=False)
        print(frame)
