# trace_alignment.py
import networkx as nx
from networkx.algorithms import isomorphism
from explaination.visualizations.utils import normalize_activity_name


def build_trace_graph(act_names, times, delta_time_seconds=3600):
    """
    根据活动名称和时间构建轨迹图（有向，含并行特性）。
    返回 DiGraph。
    """
    G = nx.DiGraph()
    n = len(act_names)
    # 添加节点，属性 value
    for i, name in enumerate(act_names):
        G.add_node(i, value=name, parallel=False)

    # 添加顺序边（根据时间差判定并行）
    for i in range(n - 1):
        # 计算绝对时间差（times 是相对小时，需乘以3600转为秒）
        time_diff = abs(times[i + 1] - times[i]) * 3600
        if time_diff <= delta_time_seconds:
            # 并行：不添加顺序边，但标记为并行
            G.nodes[i]['parallel'] = True
            G.nodes[i + 1]['parallel'] = True
        else:
            G.add_edge(i, i + 1, eventually=False)

    # 处理并行块：如果多个节点被标记为并行，它们之间不应有序边，但在你的模式匹配中，
    # 我们仅需通过节点值匹配，图结构（边）反映了顺序/并行。
    return G


def pattern_is_contained(trace_graph, pattern_graph):
    """
    使用子图同构检查 pattern_graph 是否与 trace_graph 的某个子图匹配。
    只匹配节点属性 'value'，忽略 'parallel' 和颜色等。
    """
    node_match = isomorphism.categorical_node_match('value', '')
    matcher = isomorphism.DiGraphMatcher(trace_graph, pattern_graph, node_match=node_match)
    return matcher.subgraph_is_isomorphic()


def does_trace_contain_pattern_advanced(act_names, times_hours, pattern_graph, delta_time_seconds=3600):
    """判断给定的活动名称序列+时间是否包含指定模式图。"""
    act_names = [normalize_activity_name(n) for n in act_names if n is not None]
    if len(act_names) == 0:
        return False

    # 前缀预滤: 模式的所有活动是否都在轨迹中? (O(k) vs VF2 O(n²))
    pattern_acts = set(
        pattern_graph.nodes[n].get('value', '')
        for n in pattern_graph.nodes()
    )
    if not pattern_acts.issubset(set(act_names)):
        return False  # ~80% 调用在此终止, 避免昂贵 VF2

    times_hours = times_hours[:len(act_names)]
    trace_graph = build_trace_graph(act_names, times_hours, delta_time_seconds)
    return pattern_is_contained(trace_graph, pattern_graph)
