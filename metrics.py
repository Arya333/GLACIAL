import networkx


def graph_f1(pred, true):
    assert type(pred) == type(true), f'{type(pred)=} != {type(true)=}'
    assert isinstance(pred, networkx.DiGraph) or isinstance(pred, networkx.Graph)
    assert pred.nodes() == true.nodes(), f'graph with different nodes, double-check\n{list(pred.nodes())}\n{list(true.nodes())}'

    def div(num, den):
        return num/den if den > 0 else 0

    def f1(pre, rec):
        if pre > 0 or rec > 0:
            return 2*pre*rec / (pre + rec)
        return 0

    ret = {'full_TP': 0}
    for u, v in pred.edges:
        direction = pred.edges[u, v].get('direction', '')
        if direction in {'-->', 'o->'} and true.has_edge(u, v):
            ret['full_TP'] += 1

    ret['full_pre'] = div(ret['full_TP'], len(pred.edges))
    ret['full_rec'] = div(ret['full_TP'], len(true.edges))
    ret['full_f1'] = f1(ret['full_pre'], ret['full_rec'])

    return ret


def score(pred, true, label=None):
    out = {'model': label} if label is not None else {}

    reduced = networkx.DiGraph()
    reduced.add_nodes_from(pred.nodes)
    for u, v in pred.edges:
        direction = pred.edges[u, v].get('direction', '')
        if direction == '<--':
            reduced.add_edge(v, u, direction='-->')
        elif direction in '<-o':
            reduced.add_edge(v, u, direction='o->')
        else:
            reduced.add_edge(u, v, direction=direction)

    assert len(reduced.to_undirected(reciprocal=True).edges) == 0, 'bidirectional!  check logic'

    out.update(graph_f1(reduced, true))

    return out


if __name__ == '__main__':
    pred = networkx.DiGraph()
    pred.add_nodes_from([0, 1, 2, 3])
    pred.add_edge(0, 1, direction='-->')
    pred.add_edge(2, 1, direction='-->')

    true = networkx.DiGraph()
    true.add_nodes_from([0, 1, 2, 3])
    true.add_edge(0, 1, direction='-->')
    true.add_edge(1, 2, direction='-->')
    true.add_edge(1, 3, direction='-->')
    print(graph_f1(pred, true))
