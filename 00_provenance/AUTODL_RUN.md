# AutoDL：冻结七任务正式对比

同步压缩包即可，无需逐目录手工挑文件。包中包含正式入口及依赖代码、七个 full/seed42 原协议、原模型和 encoder 工件、原始日志和模式缓存中被协议校验引用的文件、Buliga 官方源文件及许可证、已接受的隔离参考和 Search-DFA 图、Audit 规则、已有 P6-A/P6-B 验收链及必要配置。历史源码记录仍保留其原始冻结哈希；当前运行代码由后续已验收协议的指纹约束。

不包含 Windows 安装库、整套历史实验结果、PPT/Word、IDE 配置或正在生成的本地正式矩阵。包内可能含被验收链引用的历史候选，这是证据依赖，不在服务器上重新生成。

服务器使用 Python 3.12、CPU torch 2.4.1 及 configs/autodl_requirements.txt 中的当前依赖。根 requirements.txt 是旧环境导出，包含 Conda 和 Windows 特有包，不用于这次 Linux 安装。当前算法入口为 CPU 单线程；此部署不新增 GPU 或多核搜索适配。

## 本地上传

在 Windows PowerShell 中执行，将端口与主机替换为 AutoDL 实例 SSH 信息：

```powershell
scp -P PORT "D:\MMOE_UWL_cf\outputs\p6b_autodl_bundle_20260918_v2.tar.gz" root@HOST:/root/autodl-tmp/
```

也可通过 JupyterLab 或 FileZilla 上传同一个压缩包。不要上传本地 .virtualenvs 或 .runtime/p0p1_torch 中的 Windows 二进制。

## 服务器安装与路径迁移

在 AutoDL 终端逐段执行。下面路径应为空，首次执行时不会覆盖已有目录。

```bash
cd /root/autodl-tmp
tar -xzf p6b_autodl_bundle_20260918_v2.tar.gz
source /root/miniconda3/etc/profile.d/conda.sh
conda create -n cf_compare python=3.12 -y
conda activate cf_compare
python -m pip install torch==2.4.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r /root/autodl-tmp/MMOE_UWL_cf/configs/autodl_requirements.txt
cd /root/autodl-tmp/MMOE_UWL_cf
python -B tools/autodl_transfer.py relocate --destination /root/autodl-tmp/MMOE_UWL_cf_linux
cd /root/autodl-tmp/MMOE_UWL_cf_linux
```

原包中的 Windows 绝对路径不能直接被 Linux 入口读取。relocate 在独立新目录创建服务器副本，只改运行元数据的工作区路径，并逐层更新引用文件哈希及封存哈希；原模型文件、encoder 和日志的字节不变，模型配置中的历史路径保持原值以匹配 checkpoint.cfg。查询、预算、目标、规则、干预范围、种子、共同队列和统计定义均不变。原提取目录保持只读；不要对全部 JSON/模型配置做搜索替换。

本工程本轮 Audit 使用 flloat/pythomata 在 Python 中编译，无需额外安装 MONA，也无需运行训练或重新准备 Search-DFA。

## 一次运行完整流程

```bash
nohup bash tools/run_autodl_comparison.sh > outputs/autodl_pipeline.log 2>&1 &
tail -f outputs/autodl_pipeline.log
```

按生成→保存后 Audit→汇总→最终独立验收→原统计入口→CSV/Markdown 制表连续执行。任一命令非零退出即停止后续阶段。CREATED 保持原协议的不可用占位，不参与实测表格。最终验收会重评分全部保存物理池，其成本与搜索分开，不重训、不按 Audit 调参。

不要在原始 Windows 路径包目录启动实验，应在 MMOE_UWL_cf_linux 中运行。目录名固定为首次服务器运行；中断后原入口不自动续跑或覆盖，避免把两次墙钟执行混成一次。服务器速度和墙钟截断可能与本地不同，正式比较以该服务器同一轮完整矩阵为准，不混入本地部分结果。

最终输出：

- outputs/p6b_formal_autodl_v1：完整生成、Audit、指标、independent_acceptance.json 与统计输入清单。
- outputs/p6b_statistics_autodl_v1：查询指标及配对案例块置信区间。
- outputs/p6b_tables_autodl_v1：all_methods.csv、fixed_time_search_main.csv、same_pool_minimal_relaxation.csv、fixed_time_privileged_head_ablation.csv、time_intervention_auxiliary.csv、paired_intervals.csv、query_costs.csv、costs.json、RESULTS.md 和 tables_manifest.json。

本地仍在运行的矩阵不会被上述上传或服务器命令停止；若决定服务器作为正式结果来源，再明确停止本地并保留其工件即可。

## 下载结果

服务器压缩结果：

```bash
cd /root/autodl-tmp/MMOE_UWL_cf_linux
tar -czf /root/autodl-tmp/p6b_autodl_results.tar.gz outputs/p6b_formal_autodl_v1 outputs/p6b_statistics_autodl_v1 outputs/p6b_tables_autodl_v1 outputs/autodl_relocation.json outputs/autodl_pipeline.log
```

Windows 下载：

```powershell
scp -P PORT root@HOST:/root/autodl-tmp/p6b_autodl_results.tar.gz "D:\MMOE_UWL_cf\outputs\"
```

AutoDL 上传命令与数据盘路径依据 https://www.autodl.com/docs/scp/ 和 https://www.autodl.com/docs/pycharm/。本包未通过 SSH 部署或在 Linux 服务器上启动；本地路径迁移、只读协议检查等验证不等同于服务器环境已经安装或实际运行成功。
