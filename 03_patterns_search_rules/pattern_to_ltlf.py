import networkx as nx


def _repeat_next(expr, steps):
    wrapped = expr
    for _ in range(steps):
        wrapped = f"X ({wrapped})"
    return wrapped


def _bounded_future(expr, max_gap):
    horizon = max(0, int(max_gap)) + 1
    clauses = [_repeat_next(expr, steps) for steps in range(1, horizon + 1)]
    if len(clauses) == 1:
        return clauses[0]
    return "(" + " | ".join(clauses) + ")"


def _build_ordered_formula(nodes, values, edge_lookup):
    if not nodes:
        return None

    expr = values[nodes[-1]]
    for idx in range(len(nodes) - 2, -1, -1):
        u = nodes[idx]
        v = nodes[idx + 1]
        edge = edge_lookup.get((u, v), {})
        if edge.get("eventually", False):
            max_gap = edge.get("max_gap")
            if max_gap is not None:
                expr = f"({values[u]} & {_bounded_future(expr, max_gap)})"
            else:
                expr = f"({values[u]} & F {expr})"
        else:
            expr = f"({values[u]} & X {expr})"
    return f"F {expr}"


def pattern_to_ltlf(pattern_graph):
    """Translate a pattern graph into a single LTLf formula."""
    nodes = list(pattern_graph.nodes())
    if len(nodes) == 0:
        return None

    values = nx.get_node_attributes(pattern_graph, "value")
    if len(nodes) == 1:
        return f"F {values[nodes[0]]}"

    edges = list(pattern_graph.edges(data=True))
    edge_lookup = {(u, v): data for u, v, data in edges}
    sorted_nodes = sorted(nodes)

    is_linear = True
    for i in range(len(sorted_nodes) - 1):
        if (sorted_nodes[i], sorted_nodes[i + 1]) not in edge_lookup:
            is_linear = False
            break

    if is_linear:
        return _build_ordered_formula(sorted_nodes, values, edge_lookup)

    acts = [values[n] for n in sorted_nodes]
    return " & ".join([f"F {a}" for a in acts])


def extract_ltlf_from_patterns(pattern_dict, act_encoder=None):
    """Generate a conjunction of LTLf formulas from all pattern graphs."""
    sub_formulas = []
    for _, graph in pattern_dict.items():
        formula = pattern_to_ltlf(graph)
        if formula:
            sub_formulas.append(f"({formula})")
    if not sub_formulas:
        return None
    return " & ".join(sub_formulas)
