# CMT-CF

CMT-CF 用于预测性流程监控中的反事实解释。模型以流程实例的已观测事件前缀为输入，预测下一活动、下一活动发生时间和流程结果。反事实搜索对前缀中的活动做有限修改，寻找能够达到目标预测结果且满足搜索规则的替代轨迹。正式实验在候选保存后，再用独立的流程规则（Audit）检查其合规性。

对应论文：*Compliance-Constrained Counterfactual Generation for Predictive Process Monitoring with Multiple Perspectives*。

## 快速开始

以下命令适用于 Windows PowerShell。需要 Python 3.12 和至少一个已标注的事件日志。示例使用 CPU 版 PyTorch，从训练开始运行单任务实验。

### 1. 准备运行目录

项目按功能存放源码，运行时需要按[文件清单](FILE_INVENTORY.csv)恢复 Python 模块的相对路径。下面的命令将源码和配置复制到一个新目录。将首行路径替换为自己的项目目录。

```powershell
Set-Location -LiteralPath 'D:\CMT-CF'
$packageRoot = (Get-Location).Path
$runRoot = Join-Path $env:USERPROFILE 'CMT-CF-run'

if (Test-Path -LiteralPath $runRoot) {
    throw "请选择一个尚不存在的运行目录：$runRoot"
}

$rows = Import-Csv -LiteralPath (Join-Path $packageRoot 'FILE_INVENTORY.csv') -Encoding UTF8
$rows | Where-Object {
    $_.restore_original_layout -eq 'True' -and
    $_.original_path -match '^(?!\.runtime/).+\.py$|^configs/'
} | ForEach-Object {
    $target = Join-Path $runRoot $_.original_path
    New-Item -ItemType Directory -Force -Path (Split-Path -Parent $target) | Out-Null
    Copy-Item -LiteralPath (Join-Path $packageRoot $_.package_path) -Destination $target -ErrorAction Stop
}

Set-Location -LiteralPath $runRoot
```

### 2. 安装依赖

运行 `python --version` 确认当前命令使用 Python 3.12。以下命令在项目运行目录中创建独立环境。若电脑安装了多个 Python 版本，可用 Python 3.12 的完整路径执行第一条命令。

```powershell
python -m venv .venv
$python = Join-Path (Get-Location).Path '.venv\Scripts\python.exe'
& $python -m pip install torch==2.4.1 --index-url https://download.pytorch.org/whl/cpu
& $python -m pip install -r configs/autodl_requirements.txt lxml==6.1.1
& $python experiment_runner.py --help
```

### 3. 准备事件日志

论文使用 [Sepsis Cases](https://doi.org/10.4121/uuid:915d2bfb-7e84-49ad-a286-dc35f063a460) 和 [BPI Challenge 2015](https://doi.org/10.4121/uuid:31a308ef-c844-48da-948c-305d167a0ec1) 两个公开日志。论文实验按下表使用已标注的任务文件。如果持有完整实验归档，可从其中的 `logs/` 获取这些文件。自行准备数据时，应按对应任务定义生成案例结果标签，并保留事件时间和实验所使用的属性。

| 数据来源 | 任务 ID | 实验输入示例（相对于运行目录） |
| --- | --- | --- |
| Sepsis Cases | `sepsis_cases_1`、`sepsis_cases_2`、`sepsis_cases_4` | `logs/sepsis_cases_1/sepsis_cases_1.csv`，其余任务按各自 ID 命名 |
| BPI Challenge 2015 | `BPIC15_2_f2`、`BPIC15_3_f2`、`BPIC15_4_f2`、`BPIC15_5_f2` | `logs/BPIC15_2_f2/100.csv`，其余任务放在各自目录 |

入口支持 `.csv` 和 `.xes`。训练所需的核心字段为案例 ID（如 `Case ID`）、活动（如 `Activity`）、时间戳（如 `time:timestamp`）和案例级结果 `label`。任务的标签映射及静态、动态属性见 [dataset_confs.py](02_data_preprocessing/dataset_confs.py)。使用下载的原始日志时，请先检查 `label` 是否已经按目标任务生成。

### 4. 运行训练和反事实评估

将示例的 Sepsis 文件路径替换为实际路径。下面命令用一个训练轮次和少量前缀检查数据加载、模型训练及反事实评估是否能运行。

```powershell
& $python -u experiment_runner.py `
    --datasets sepsis_cases_1 --variants full --seeds 42 `
    --data_path 'logs/sepsis_cases_1/sepsis_cases_1.csv' `
    --epochs 1 --batch_size 64 --num_workers 0 `
    --max_prefixes_per_case 2 --cf_eval_samples 5 `
    --output_dir results/quickstart
```

完整训练可增加 `--epochs`，省略 `--max_prefixes_per_case` 以使用全部合格前缀，并设定 `--cf_eval_samples 0` 评估全部符合条件的测试前缀。批量运行时，可在 `--datasets` 后列出多个任务，将对应文件放在各自的 `logs/<任务ID>/` 目录，省略 `--data_path`。单任务测试建议显式指定文件，以避免自动查找选中同目录下的其他日志。

结果默认保存在 `--output_dir` 加时间戳的目录。训练检查点命名为 `model_<任务>_<变体>_seed<种子>.pt`。`all_results.csv` 汇总实验指标，`predictions_*.csv` 记录逐前缀预测，生成候选时可查看 `cf_details_*.csv`。规则与查询清单默认分别写入 `audit_manifests/` 和 `evaluation_manifests/`。某个任务失败时，查看输出目录中的 `failures.csv` 和终端信息。

## 常用参数

下表列出 `experiment_runner.py` 的常用参数。其余选项可用 `--help` 查看。

| 参数 | 默认值 | 用途 |
| --- | --- | --- |
| `--datasets` | 论文使用的七个任务 | 指定一个或多个任务 ID |
| `--data_path` | 自动在 `logs/` 查找 | 单任务运行时明确指定数据文件 |
| `--variants` | 多个模型变体 | `full` 运行完整模型。`no_pattern`、`no_uwl`、`no_dfa` 可用于消融 |
| `--seeds`、`--split_seed` | `17 42 101 2026 4096`、`42` | 分别设置模型训练种子和案例划分种子 |
| `--epochs`、`--batch_size`、`--learning_rate` | `15`、`64`、`0.001` | 设置训练轮次、批大小和学习率 |
| `--num_workers` | `2` | 设置数据加载进程数。Windows 可用 `0` 单进程加载 |
| `--prefix_mode`、`--max_prefixes_per_case` | `all`、不设上限 | 控制每个案例使用的前缀。`last` 仅取最后一个前缀 |
| `--constraint_policy` | `calibrated` | 通过校准划分硬约束和软模式。`fixed` 用于固定阈值的消融 |
| `--audit_rule_source`、`--audit_rule_file` | `heldout`、无 | 指定独立 Audit 规则来源。`external` 和 `hybrid` 需提供规则文件 |
| `--cf_eval_samples` | `0` | 设置测试前缀评估数量，`0` 表示评估全部符合条件的前缀 |
| `--cf_topk`、`--cf_intervention_points` | `5`、`3` | 设置每个查询的最大返回数和搜索时考虑的候选修改位置数 |
| `--cf_per_position_limit`、`--cf_combination_budget` | `5`、`100` | 设置每个位置保留的候选动作数和每个前缀的组合评估上限 |
| `--cf_ranking` | `pareto` | 选择候选排序方式，可设为 `pareto`、`score` 或 `dpp` |
| `--cf_max_relaxed_constraints` | 不设单独上限 | 设置搜索可豁免的硬规则数上限。设为 `0` 时禁止豁免 |
| `--output_dir`、`--no-timestamp` | `./results`、未启用 | 指定结果目录。启用 `--no-timestamp` 时直接在指定目录写入 |

模型结构见 [config.py](01_prediction_training/config.py) 和 [models.py](01_prediction_training/models.py)。共享 Transformer 的隐藏维度为 128，使用 2 层编码器和 4 个注意力头。层数、隐藏维度等由模型配置管理。

## 复现论文中的冻结实验

正式对比使用七个任务的 `full`、seed 42 冻结检查点。其配置、查询、目标和搜索预算由冻结协议确定。正式主方法是 `strict_dfa_outcome`，每个查询与方法最多使用 500 个模型评估单位、10,000 次候选尝试及 60 秒的搜索运行时间，最多返回 5 个候选。训练入口的 `--cf_*` 参数适用于快速开始中的训练与评估流程。

从头重跑正式对比需要完整的 `p6b_autodl_bundle_20260918_v2.tar.gz`，其中包含与冻结协议匹配的数据和依赖文件。拿到归档后，在 Linux 或 AutoDL 的新工作目录执行以下命令。

```bash
tar -xzf p6b_autodl_bundle_20260918_v2.tar.gz
cd MMOE_UWL_cf
python3.12 -m venv ../cmtcf-venv
source ../cmtcf-venv/bin/activate
python -m pip install torch==2.4.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r configs/autodl_requirements.txt lxml==6.1.1
python -B tools/autodl_transfer.py relocate --destination ../CMT-CF-formal
cd ../CMT-CF-formal
bash tools/run_autodl_comparison.sh
```

迁移脚本将冻结协议中的路径和关联校验值更新到新的运行目录。流水线依次生成候选、执行独立 Audit、汇总及验收结果，并输出统计与表格。每次从新运行目录启动完整流水线。

| 参数或子命令 | 用法 |
| --- | --- |
| `validate --config <配置>` | 检查桥接配置。默认使用 `configs/cf_p6b_bridge.json` |
| `search --preflight-dir <目录> --run-dir <目录> --authorization <JSON>` | 生成反事实候选。使用迁移脚本生成的授权文件 |
| `audit`、`summarize`、`verify` + `--run-dir <目录>` | 依次审核、汇总、独立验收已保存的候选 |
| `statistics --preflight-dir <目录> --run-dir <目录> --output-dir <目录>` | 生成配对统计结果 |

上述子命令由 [run_cf_formal_bridge.py](06_experiment_and_baseline/run_cf_formal_bridge.py) 提供。正式候选、规则审核及指标保存在 `outputs/p6b_formal_autodl_v1/`，统计数据保存在 `outputs/p6b_statistics_autodl_v1/`，汇总表保存在 `outputs/p6b_tables_autodl_v1/`。部署及输出说明见 [AUTODL_RUN.md](00_provenance/AUTODL_RUN.md)。

## 目录与现有结果

| 目录 | 内容 |
| --- | --- |
| `00_provenance/` | 实验来源、运行记录和工件校验信息 |
| `01_prediction_training/` | 多任务模型、训练入口和配置 |
| `02_data_preprocessing/` | 日志读取、前缀构造、属性编码和任务定义 |
| `03_patterns_search_rules/` | 模式挖掘及搜索规则转换 |
| `04_counterfactual_search/` | 反事实生成、搜索预算和正式实验桥接 |
| `05_audit_evaluation/` | 独立 Audit 规则、候选评分和统计 |
| `06_experiment_and_baseline/` | 正式实验入口、比较方法及环境依赖 |
| `07_frozen_models_and_protocols/` | 按任务归档的模型权重、冻结协议和预测上下文 |
| `08_paper_evidence/` | 论文表格和图的结果依据 |

- 表 3 的历史消融见 [all_results.csv](08_paper_evidence/Table3_legacy_ablation/all_results.csv)。
- 表 4–5 和图 6–9 的正式比较见 [RESULTS.md](08_paper_evidence/Tables4_5_and_Figs6_9/tables/RESULTS.md) 和 [fixed_time_search_main.csv](08_paper_evidence/Tables4_5_and_Figs6_9/tables/fixed_time_search_main.csv)。
- 表 6 的多任务预测对比见 [summary.json](08_paper_evidence/Table6_multitask_prediction/summary.json) 及对应的 [candidate_pairs.csv](08_paper_evidence/Table6_multitask_prediction/candidate_pairs.csv)。

数据集的引用信息见各自的数据页面。Buliga 基线的上游许可见 [LICENSE.md](06_experiment_and_baseline/upstream/buliga/LICENSE.md)。
