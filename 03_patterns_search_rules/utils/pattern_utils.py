import networkx as nx

def trace_contains_pattern(act_seq, pattern_graph):
    """检查 act_seq 是否按顺序包含模式图中的所有活动节点"""
    nodes = sorted(pattern_graph.nodes())
    values = nx.get_node_attributes(pattern_graph, 'value')
    pattern_acts = [values[n] for n in nodes]
    it = iter(act_seq)
    return all(p_act in it for p_act in pattern_acts)