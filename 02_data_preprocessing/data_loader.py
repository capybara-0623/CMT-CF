import os
import pandas as pd
import numpy as np
from lxml import etree
from dataset_confs import DatasetConfs


def _find_dataset_file(dataset_name):
    """在 logs 目录中查找数据集文件，支持子目录搜索和多种命名模式"""
    logs_dir = 'logs'

    # 候选文件名列表（按优先级）
    candidates = []

    # 直接在 logs/ 下
    for ext in ['.csv', '.xes']:
        candidates.append(os.path.join(logs_dir, dataset_name + ext))

    # 在 logs/<dataset_name>/ 子目录中 (优先 CSV)
    subdir = os.path.join(logs_dir, dataset_name)
    if os.path.isdir(subdir):
        for fname in sorted(os.listdir(subdir)):
            if fname.endswith('.csv') or fname.endswith('.xes'):
                candidates.append(os.path.join(subdir, fname))

        # 常见的命名模式 (CSV 优先)
        for fname in ['full.csv', dataset_name + '.csv', 'full.xes', dataset_name + '.xes']:
            p = os.path.join(subdir, fname)
            if p not in candidates:
                candidates.append(p)

    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate
    return None


def _read_csv_smart(file_path):
    """智能读取 CSV：自动检测分隔符（先 ; 后 ,）"""
    # 尝试分号优先（BPIC/Sepsis 数据集标准格式）
    try:
        df = pd.read_csv(file_path, sep=';')
        if df.shape[1] > 1:
            return df
    except Exception:
        pass

    # 尝试逗号
    try:
        df = pd.read_csv(file_path, sep=',')
        if df.shape[1] > 1:
            return df
    except Exception:
        pass

    # Python 引擎自动检测
    try:
        return pd.read_csv(file_path, sep=None, engine='python')
    except Exception:
        pass

    # 最终用 C 引擎读（允许行数不一致）
    return pd.read_csv(file_path, on_bad_lines='skip')


def parse_xes_to_dataframe(file_path):
    """
    使用 lxml 解析 XES 文件，返回 DataFrame。
    自动处理 trace 级别的属性（如 concept:name 作为 Case ID、label 等）
    和 event 级别的属性。

    输出列名映射:
      - trace 级 concept:name → Case ID
      - event 级 concept:name → Activity
      - 其他属性保留原名（如 time:timestamp, org:resource, label 等）
    """
    tree = etree.parse(file_path)
    root = tree.getroot()

    # 尝试常见命名空间或无命名空间
    ns = {'xes': 'http://www.xes-standard.org/'}
    traces = root.findall('.//xes:trace', ns) or root.findall('.//trace')
    if not traces:
        traces = root.findall('.//trace')

    rows = []
    for trace in traces:
        # 提取 trace 级别的属性
        trace_attrs = {}
        for child in trace:
            tag = child.tag.split('}')[-1] if '}' in child.tag else child.tag
            if tag in ('string', 'date', 'int', 'float'):
                key = child.get('key')
                value = child.get('value')
                if key:
                    trace_attrs[key] = value

        # 处理事件
        events = trace.findall('.//event') or trace.findall('event')
        for event in events:
            event_attrs = {}
            for attr in event:
                tag = attr.tag.split('}')[-1] if '}' in attr.tag else attr.tag
                if tag in ('string', 'date', 'int', 'float'):
                    key = attr.get('key')
                    value = attr.get('value')
                    if key:
                        event_attrs[key] = value
            rows.append(event_attrs)

    df = pd.DataFrame(rows)

    # 如果 trace 有 concept:name，将其作为 Case ID 合并到每个事件
    # （由于上面已经提取了 event 属性，但 trace 属性需要从 trace 级别复制）
    if len(rows) == 0:
        return df

    # 重新解析：trace 属性附加到每个事件
    rows2 = []
    for trace in traces:
        # 提取 trace 属性
        trace_kv = {}
        for child in trace:
            tag = child.tag.split('}')[-1] if '}' in child.tag else child.tag
            if tag in ('string', 'date', 'int', 'float'):
                key = child.get('key')
                value = child.get('value')
                if key:
                    trace_kv[key] = value

        events = trace.findall('.//event') or trace.findall('event')
        for event in events:
            event_kv = dict(trace_kv)  # 继承 trace 属性
            for attr in event:
                tag = attr.tag.split('}')[-1] if '}' in attr.tag else attr.tag
                if tag in ('string', 'date', 'int', 'float'):
                    key = attr.get('key')
                    value = attr.get('value')
                    if key:
                        event_kv[key] = value
            rows2.append(event_kv)

    df = pd.DataFrame(rows2)
    if df.empty:
        return df

    # 列名映射：将 XES 标准名映射到 DatasetConfs 期望的名称
    # concept:name at trace level → Case ID (already included via trace_kv)
    # concept:name at event level → Activity (event overwrites trace value)
    # But both trace and event have concept:name — the event value wins
    if 'concept:name' in df.columns:
        # 如果已有 Case ID 列（来自 trace attribute copy），event 的 concept:name
        # 实际上覆盖了 trace 的。我们需要区分两者。
        # 简单方案：创建 Case ID 从 trace，Activity 从 event
        # 由于 event 解析覆盖了 trace 的同名属性，
        # 我们需要在解析时分别保存
        pass

    # 对没有 Case ID 列的情况，尝试从 concept:name 推断
    # 这在 XES 中比较棘手，因为 concept:name 同时用于 trace 和 event
    # 我们采用更稳健的方法：回到解析阶段

    # 第三次解析（最终版）：明确区分 trace 和 event 属性
    rows3 = []
    for trace in traces:
        # trace 属性
        case_id = None
        trace_label = None
        extra_trace_attrs = {}
        for child in trace:
            tag = child.tag.split('}')[-1] if '}' in child.tag else child.tag
            if tag in ('string', 'date', 'int', 'float'):
                key = child.get('key')
                value = child.get('value')
                if key == 'concept:name':
                    case_id = value
                elif key == 'label':
                    trace_label = value
                elif key:
                    extra_trace_attrs[key] = value

        events = trace.findall('.//event') or trace.findall('event')
        for event in events:
            event_kv = {}
            if case_id is not None:
                event_kv['Case ID'] = case_id
            if trace_label is not None:
                event_kv['label'] = trace_label
            event_kv.update(extra_trace_attrs)

            for attr in event:
                tag = attr.tag.split('}')[-1] if '}' in attr.tag else attr.tag
                if tag in ('string', 'date', 'int', 'float'):
                    key = attr.get('key')
                    value = attr.get('value')
                    if key == 'concept:name':
                        event_kv['Activity'] = value
                    elif key:
                        event_kv[key] = value
            rows3.append(event_kv)

    return pd.DataFrame(rows3)


def _apply_bp17_anti_leak(dataset_name, dynamic_cat_cols, enabled):
    removed = []
    if enabled and dataset_name.startswith("BPIC17"):
        blocked = {"accepted", "selected"}
        kept = []
        for col in dynamic_cat_cols:
            if str(col).strip().lower() in blocked:
                removed.append(col)
            else:
                kept.append(col)
        dynamic_cat_cols = kept
    return dynamic_cat_cols, removed


def _drop_target_proxy_columns(col_list, blocked_names):
    kept, removed = [], []
    blocked = {str(c).strip().lower() for c in blocked_names if c}
    for col in col_list:
        if str(col).strip().lower() in blocked:
            removed.append(col)
        else:
            kept.append(col)
    return kept, removed


def load_public_dataset(dataset_name: str, data_path: str = None, bp17_anti_leak: bool = True):
    """加载公开数据集，返回样本列表。"""
    # 查找文件
    if data_path:
        file_path = data_path
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"指定的文件不存在: {data_path}")
    else:
        # 先尝试 DatasetConfs 中预设的路径
        conf_temp = DatasetConfs(dataset_name)
        file_path = conf_temp.filename.get(dataset_name, '')
        if not file_path or not os.path.exists(file_path):
            # 在 logs 目录及子目录中搜索
            file_path = _find_dataset_file(dataset_name)

    if not file_path or not os.path.exists(file_path):
        raise FileNotFoundError(
            f"无法找到数据集 '{dataset_name}' 的文件。\n"
            f"  请将文件放在 logs/ 目录中，或使用 --data_path 指定路径。\n"
            f"  查找路径: logs/{dataset_name}.csv, logs/{dataset_name}.xes, "
            f"logs/{dataset_name}/full.csv, logs/{dataset_name}/full.xes"
        )

    print(f"找到数据集文件: {file_path}")

    # 读取数据
    if file_path.endswith('.csv'):
        df = _read_csv_smart(file_path)
    elif file_path.endswith('.xes'):
        df = parse_xes_to_dataframe(file_path)
    else:
        raise ValueError("仅支持 .csv 或 .xes 格式。")

    if df.empty:
        raise ValueError(f"数据集文件 {file_path} 解析后为空。")

    print(f"数据维度: {df.shape}, 列: {list(df.columns)[:10]}...")

    # 获取配置
    conf = DatasetConfs(dataset_name)
    case_col = conf.case_id_col[dataset_name]
    act_col = conf.activity_col[dataset_name]
    time_col = conf.timestamp_col[dataset_name]
    label_col = conf.label_col[dataset_name]
    pos_label = conf.pos_label[dataset_name]
    neg_label = conf.neg_label[dataset_name]

    # 检查必要列是否存在，若缺失尝试自动推断
    if case_col not in df.columns:
        # 尝试常见 case ID 列名
        for alt in ['Case ID', 'case:concept:name', 'case_id', 'CaseId']:
            if alt in df.columns:
                case_col = alt
                print(f"Case ID 列: 使用 '{alt}' 替代")
                break
        else:
            raise KeyError(f"找不到 Case ID 列 (试过 '{case_col}')。可用列: {list(df.columns)}")

    if act_col not in df.columns:
        for alt in ['Activity', 'concept:name', 'activity']:
            if alt in df.columns:
                act_col = alt
                print(f"Activity 列: 使用 '{alt}' 替代")
                break
        else:
            raise KeyError(f"找不到 Activity 列 (试过 '{act_col}')。可用列: {list(df.columns)}")

    if time_col not in df.columns:
        for alt in ['time:timestamp', 'timestamp', 'Complete Timestamp']:
            if alt in df.columns:
                time_col = alt
                print(f"Timestamp 列: 使用 '{alt}' 替代")
                break
        else:
            raise KeyError(f"找不到 Timestamp 列 (试过 '{time_col}')。可用列: {list(df.columns)}")

    label_available = label_col in df.columns
    if not label_available:
        print(f"警告: 标签列 '{label_col}' 不存在，将设置 outcome=-1（无监督模式）")

    # 自动推断正例标签：当数据中的标签值与配置的 pos_label 不匹配时，
    # 自动检测 true/1/deviant 等常见正例表示
    if label_available:
        actual_labels = set(str(v) for v in df[label_col].dropna().unique())
        if str(pos_label) not in actual_labels:
            # 尝试常见正例标签映射
            pos_candidates = ['true', 'True', '1', 'deviant', 'positive', 'yes', 'accepted']
            detected_pos = None
            for c in pos_candidates:
                if c in actual_labels:
                    detected_pos = c
                    break
            if detected_pos is not None:
                print(f"标签映射: 配置 pos_label=\"{pos_label}\" 不在数据中，"
                      f"自动检测正例标签为 \"{detected_pos}\"")
                pos_label = detected_pos
            else:
                print(f"警告: 配置的 pos_label=\"{pos_label}\" 不在数据标签 {actual_labels} 中，"
                      f"将使用原始值比较")

    # 转换时间戳
    df[time_col] = pd.to_datetime(df[time_col], utc=True)

    # ---- 提取属性列 ----
    static_cat_cols = conf.static_cat_cols.get(dataset_name, [])
    static_num_cols = conf.static_num_cols.get(dataset_name, [])
    dynamic_cat_cols = conf.dynamic_cat_cols.get(dataset_name, [])
    dynamic_num_cols = conf.dynamic_num_cols.get(dataset_name, [])
    blocked_attr_cols = {label_col, "label", "outcome", "target"}
    static_cat_cols, removed_static_cat_targets = _drop_target_proxy_columns(static_cat_cols, blocked_attr_cols)
    static_num_cols, removed_static_num_targets = _drop_target_proxy_columns(static_num_cols, blocked_attr_cols)
    dynamic_cat_cols, removed_dynamic_cat_targets = _drop_target_proxy_columns(dynamic_cat_cols, blocked_attr_cols)
    dynamic_num_cols, removed_dynamic_num_targets = _drop_target_proxy_columns(dynamic_num_cols, blocked_attr_cols)
    removed_target_cols = (
        removed_static_cat_targets + removed_static_num_targets +
        removed_dynamic_cat_targets + removed_dynamic_num_targets
    )
    if removed_target_cols:
        print("  [anti-leak] removed target/proxy attribute columns: "
              + ", ".join(removed_target_cols))
    dynamic_cat_cols, removed_bp17_cols = _apply_bp17_anti_leak(
        dataset_name, dynamic_cat_cols, bp17_anti_leak
    )
    raw_static_cat_cols = [c for c in static_cat_cols if c in df.columns]
    raw_static_num_cols = [c for c in static_num_cols if c in df.columns]
    raw_dynamic_cat_cols = [c for c in dynamic_cat_cols if c in df.columns]
    raw_dynamic_num_cols = [c for c in dynamic_num_cols if c in df.columns]
    if removed_bp17_cols:
        print("  [BPIC17 anti-leak] removed dynamic categorical columns: "
              + ", ".join(removed_bp17_cols))

    # 对类别属性做 Label 编码 + 低信号列过滤
    from sklearn.preprocessing import LabelEncoder
    attr_encoders = {}
    filtered_cols = {"static_cat": [], "dynamic_cat": []}

    for col_list, prefix in [
        (static_cat_cols, "static_cat"),
        (dynamic_cat_cols, "dynamic_cat"),
    ]:
        for col in col_list:
            if col not in df.columns:
                continue
            # 检查: 顶层类占比 > 90% 的列视为低信号, 跳过
            raw_vals = df[col].fillna("MISSING").astype(str)
            top_ratio = raw_vals.value_counts(normalize=True).iloc[0]
            if top_ratio > 0.90:
                filtered_cols[prefix].append(
                    f"{col}(top={top_ratio:.1%})"
                )
                continue
            le = LabelEncoder()
            le.fit(raw_vals)
            attr_encoders[f"{prefix}_{col}"] = le

    # 打印过滤信息
    for prefix in ["static_cat", "dynamic_cat"]:
        if filtered_cols[prefix]:
            print(f"  过滤低信号{prefix}: {', '.join(filtered_cols[prefix])}")

    # 重新确定有效列
    static_cat_cols = [c for c in static_cat_cols
                       if c in df.columns and f"static_cat_{c}" in attr_encoders]
    dynamic_cat_cols = [c for c in dynamic_cat_cols
                        if c in df.columns and f"dynamic_cat_{c}" in attr_encoders]

    # 数值属性归一化 (StandardScaler per column)
    from sklearn.preprocessing import StandardScaler
    num_scalers = {}
    for col_list, prefix in [(static_num_cols, "static_num"), (dynamic_num_cols, "dynamic_num")]:
        for col in col_list:
            if col in df.columns:
                scaler = StandardScaler()
                scaler.fit(df[[col]].fillna(0).values)
                num_scalers[f"{prefix}_{col}"] = scaler

    samples = []
    for case, group in df.groupby(case_col):
        group = group.sort_values(time_col)
        acts = group[act_col].tolist()
        times_abs = group[time_col].tolist()
        base = times_abs[0]
        times_rel = [(t - base).total_seconds() / 3600.0 for t in times_abs]

        # 获取标签
        if label_available:
            last_label = group[label_col].iloc[-1]
            outcome = 1 if str(last_label) == str(pos_label) else 0
        else:
            outcome = -1

        # 静态属性 (取第一行)
        static_cat_vals, static_num_vals = [], []
        for col in static_cat_cols:
            if col in group.columns:
                le = attr_encoders.get(f"static_cat_{col}")
                raw = str(group[col].iloc[0]) if not pd.isna(group[col].iloc[0]) else "MISSING"
                static_cat_vals.append(int(le.transform([raw])[0]) if le else 0)
        for col in static_num_cols:
            if col in group.columns:
                v = group[col].iloc[0]
                val = float(v) if not pd.isna(v) else 0.0
                scaler = num_scalers.get(f"static_num_{col}")
                if scaler:
                    val = float(scaler.transform([[val]])[0, 0])
                static_num_vals.append(val)

        # 动态属性 (每事件一行)
        dynamic_cat_vals, dynamic_num_vals = [], []
        for col in dynamic_cat_cols:
            col_vals = []
            if col in group.columns:
                le = attr_encoders.get(f"dynamic_cat_{col}")
                raw_vals = group[col].fillna("MISSING").astype(str).tolist()
                col_vals = [int(le.transform([v])[0]) if le else 0 for v in raw_vals]
            dynamic_cat_vals.append(col_vals)
        for col in dynamic_num_cols:
            col_vals = []
            if col in group.columns:
                vals = group[col].fillna(0).astype(float).tolist()
                scaler = num_scalers.get(f"dynamic_num_{col}")
                if scaler:
                    vals = [float(scaler.transform([[v]])[0, 0]) for v in vals]
                col_vals = vals
            dynamic_num_vals.append(col_vals)

        # 转置: 每列一组 → 每事件一组
        L = len(acts)
        dyn_cat_per_event = []
        dyn_num_per_event = []
        for t in range(L):
            cat_vec = [dynamic_cat_vals[ci][t] if ci < len(dynamic_cat_vals)
                       and t < len(dynamic_cat_vals[ci]) else 0
                       for ci in range(len(dynamic_cat_cols))]
            num_vec = [dynamic_num_vals[ci][t] if ci < len(dynamic_num_vals)
                       and t < len(dynamic_num_vals[ci]) else 0.0
                       for ci in range(len(dynamic_num_cols))]
            dyn_cat_per_event.append(cat_vec)
            dyn_num_per_event.append(num_vec)

        samples.append({
            "activities": acts,
            "times": times_rel,
            "outcome": outcome,
            "length": L,
            "case_id": str(case),
            "static_cat": static_cat_vals,
            "static_num": static_num_vals,
            "dynamic_cat": dyn_cat_per_event,
            "dynamic_num": dyn_num_per_event,
            "static_cat_raw": {
                col: (str(group[col].iloc[0]) if col in group.columns and not pd.isna(group[col].iloc[0]) else "MISSING")
                for col in raw_static_cat_cols
            },
            "static_num_raw": {
                col: (float(group[col].iloc[0]) if col in group.columns and not pd.isna(group[col].iloc[0]) else 0.0)
                for col in raw_static_num_cols
            },
            "dynamic_cat_raw": {
                col: (group[col].fillna("MISSING").astype(str).tolist() if col in group.columns else [])
                for col in raw_dynamic_cat_cols
            },
            "dynamic_num_raw": {
                col: (group[col].fillna(0).astype(float).tolist() if col in group.columns else [])
                for col in raw_dynamic_num_cols
            },
        })

    # 初始化模式特征字段
    for s in samples:
        s['pattern_features'] = []

    # ---- 时间归一化到 [0, 1] ----
    global_max_time = max(
        max(s["times"]) for s in samples if len(s["times"]) > 0
    )
    if global_max_time > 0:
        for s in samples:
            s["times"] = [t / global_max_time for t in s["times"]]
        print(f"  时间归一化: max={global_max_time:.1f}h → [0, 1]")

    # 统计属性维度
    attr_dims = {
        "n_static_cat": len(static_cat_cols),
        "n_static_num": len(static_num_cols),
        "n_dynamic_cat": len(dynamic_cat_cols),
        "n_dynamic_num": len(dynamic_num_cols),
        "static_cat_cols": list(static_cat_cols),
        "static_num_cols": list(static_num_cols),
        "dynamic_cat_cols": list(dynamic_cat_cols),
        "dynamic_num_cols": list(dynamic_num_cols),
        "raw_static_cat_cols": list(raw_static_cat_cols),
        "raw_static_num_cols": list(raw_static_num_cols),
        "raw_dynamic_cat_cols": list(raw_dynamic_cat_cols),
        "raw_dynamic_num_cols": list(raw_dynamic_num_cols),
        "bp17_anti_leak": bool(bp17_anti_leak and dataset_name.startswith("BPIC17")),
        "removed_bp17_cols": list(removed_bp17_cols),
        "attr_encoders": attr_encoders,
    }

    for s in samples:
        s['pattern_features'] = []

    print(f"加载完成: {len(samples)} 个案例, "
          f"正例率: {sum(1 for s in samples if s['outcome'] == 1) / max(len(samples), 1):.2%}")
    print(f"  属性: 静态类别{len(static_cat_cols)} 静态数值{len(static_num_cols)} "
          f"动态类别{len(dynamic_cat_cols)} 动态数值{len(dynamic_num_cols)}")

    return samples, attr_dims


def _fit_cat_mapping(values):
    uniq = sorted({str(v) if v is not None else "MISSING" for v in values})
    return {v: i + 1 for i, v in enumerate(uniq)}


def _transform_num(value, stats):
    mean, scale = stats
    scale = scale if scale > 0 else 1.0
    try:
        value = float(value)
    except Exception:
        value = 0.0
    return float((value - mean) / scale)


def fit_transform_attributes_train_only(samples, attr_dims, train_end, low_signal_threshold=0.90):
    """Refit attribute preprocessing on the train split and transform all samples.

    The initial loader keeps raw attribute values in each sample.  This function
    overwrites encoded/scaled attributes using train-only category mappings,
    train-only low-signal filtering, and train-only numeric statistics.
    """
    train_samples = samples[:train_end]
    raw_static_cat_cols = attr_dims.get("raw_static_cat_cols", attr_dims.get("static_cat_cols", []))
    raw_static_num_cols = attr_dims.get("raw_static_num_cols", attr_dims.get("static_num_cols", []))
    raw_dynamic_cat_cols = attr_dims.get("raw_dynamic_cat_cols", attr_dims.get("dynamic_cat_cols", []))
    raw_dynamic_num_cols = attr_dims.get("raw_dynamic_num_cols", attr_dims.get("dynamic_num_cols", []))

    active_static_cat_cols, active_dynamic_cat_cols = [], []
    cat_mappings = {}
    filtered_cols = {"static_cat": [], "dynamic_cat": []}

    for col in raw_static_cat_cols:
        vals = [s.get("static_cat_raw", {}).get(col, "MISSING") for s in train_samples]
        vals = [str(v) if v is not None else "MISSING" for v in vals]
        if not vals:
            continue
        top_ratio = max(vals.count(v) for v in set(vals)) / max(len(vals), 1)
        if top_ratio > low_signal_threshold:
            filtered_cols["static_cat"].append(f"{col}(top={top_ratio:.1%})")
            continue
        active_static_cat_cols.append(col)
        cat_mappings[f"static_cat_{col}"] = _fit_cat_mapping(vals)

    for col in raw_dynamic_cat_cols:
        vals = []
        for s in train_samples:
            vals.extend(s.get("dynamic_cat_raw", {}).get(col, []))
        vals = [str(v) if v is not None else "MISSING" for v in vals]
        if not vals:
            continue
        top_ratio = max(vals.count(v) for v in set(vals)) / max(len(vals), 1)
        if top_ratio > low_signal_threshold:
            filtered_cols["dynamic_cat"].append(f"{col}(top={top_ratio:.1%})")
            continue
        active_dynamic_cat_cols.append(col)
        cat_mappings[f"dynamic_cat_{col}"] = _fit_cat_mapping(vals)

    if filtered_cols["static_cat"]:
        print("  Train-only filter static_cat: " + ", ".join(filtered_cols["static_cat"]))
    if filtered_cols["dynamic_cat"]:
        print("  Train-only filter dynamic_cat: " + ", ".join(filtered_cols["dynamic_cat"]))

    num_stats = {}
    active_static_num_cols = [c for c in raw_static_num_cols]
    active_dynamic_num_cols = [c for c in raw_dynamic_num_cols]
    for col in active_static_num_cols:
        vals = [float(s.get("static_num_raw", {}).get(col, 0.0)) for s in train_samples]
        arr = np.asarray(vals if vals else [0.0], dtype=float)
        num_stats[f"static_num_{col}"] = (float(arr.mean()), float(arr.std() if arr.std() > 0 else 1.0))
    for col in active_dynamic_num_cols:
        vals = []
        for s in train_samples:
            vals.extend(s.get("dynamic_num_raw", {}).get(col, []))
        arr = np.asarray(vals if vals else [0.0], dtype=float)
        num_stats[f"dynamic_num_{col}"] = (float(arr.mean()), float(arr.std() if arr.std() > 0 else 1.0))

    for s in samples:
        L = int(s.get("length", len(s.get("activities", []))))
        s["static_cat"] = [
            cat_mappings[f"static_cat_{col}"].get(str(s.get("static_cat_raw", {}).get(col, "MISSING")), 0)
            for col in active_static_cat_cols
        ]
        s["static_num"] = [
            _transform_num(s.get("static_num_raw", {}).get(col, 0.0), num_stats[f"static_num_{col}"])
            for col in active_static_num_cols
        ]

        dyn_cat_rows = []
        for t in range(L):
            row = []
            for col in active_dynamic_cat_cols:
                raw_vals = s.get("dynamic_cat_raw", {}).get(col, [])
                raw = raw_vals[t] if t < len(raw_vals) else "MISSING"
                row.append(cat_mappings[f"dynamic_cat_{col}"].get(str(raw), 0))
            dyn_cat_rows.append(row)
        s["dynamic_cat"] = dyn_cat_rows

        dyn_num_rows = []
        for t in range(L):
            row = []
            for col in active_dynamic_num_cols:
                raw_vals = s.get("dynamic_num_raw", {}).get(col, [])
                raw = raw_vals[t] if t < len(raw_vals) else 0.0
                row.append(_transform_num(raw, num_stats[f"dynamic_num_{col}"]))
            dyn_num_rows.append(row)
        s["dynamic_num"] = dyn_num_rows

    attr_dims.update({
        "n_static_cat": len(active_static_cat_cols),
        "n_static_num": len(active_static_num_cols),
        "n_dynamic_cat": len(active_dynamic_cat_cols),
        "n_dynamic_num": len(active_dynamic_num_cols),
        "static_cat_cols": list(active_static_cat_cols),
        "static_num_cols": list(active_static_num_cols),
        "dynamic_cat_cols": list(active_dynamic_cat_cols),
        "dynamic_num_cols": list(active_dynamic_num_cols),
        "attr_encoders": cat_mappings,
        "num_scalers": num_stats,
        "train_only_preprocessing": True,
    })
    print("  Attribute preprocessing: train-only fit applied")
    return samples, attr_dims


class TrainOnlyActivityEncoder:
    def __init__(self, classes):
        self.classes_ = np.asarray(classes, dtype=object)
        self._to_index = {str(v): i for i, v in enumerate(self.classes_)}

    def transform(self, values):
        return np.asarray([self._to_index.get(str(v), 0) for v in values], dtype=int)

    def inverse_transform(self, indices):
        out = []
        for idx in indices:
            idx = int(idx)
            if 0 <= idx < len(self.classes_):
                out.append(self.classes_[idx])
            else:
                out.append("__UNK__")
        return np.asarray(out, dtype=object)


def encode_activities_train_only(samples, train_end, num_special_tokens=1):
    train_acts = []
    for s in samples[:train_end]:
        train_acts.extend(str(a) for a in s["activities"])
    known = sorted(set(train_acts))
    classes = ["__UNK__"] + [a for a in known if a != "__UNK__"]
    encoder = TrainOnlyActivityEncoder(classes)
    unknown_count = 0
    for s in samples:
        encoded = []
        for act in s["activities"]:
            idx = int(encoder.transform([str(act)])[0])
            if idx == 0 and str(act) != "__UNK__":
                unknown_count += 1
            encoded.append(idx + num_special_tokens)
        s["activities"] = encoded
    num_activities = len(encoder.classes_) + num_special_tokens
    if unknown_count:
        print(f"  Activity encoding: {unknown_count} unseen validation/test events mapped to __UNK__")
    print("  Activity encoding: train-only vocabulary applied")
    return samples, num_activities, encoder


def encode_activities(samples, num_special_tokens=1):
    from sklearn.preprocessing import LabelEncoder
    all_acts = []
    for s in samples:
        all_acts.extend(s["activities"])
    encoder = LabelEncoder()
    encoder.fit(list(set(all_acts)))
    for s in samples:
        s["activities"] = (encoder.transform(s["activities"]) + num_special_tokens).tolist()
    num_activities = len(encoder.classes_) + num_special_tokens
    return samples, num_activities, encoder
