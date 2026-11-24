# CMT-CF 论文模型与实验组件

原项目：D:/MMOE_UWL_cf  
论文：Compliance-Constrained Counterfactual Generation for Predictive Process Monitoring with Multiple Perspectives

## 论文实际使用的模型

正式反事实对比和多任务解释分析使用同一套 CMT-CF 方案下、分别针对七个任务训练的 full / seed 42 检查点。任务为 Sepsis Cases 1、2、4，以及 BPIC15 2、3、4、5 的 f2 变体。每份权重、对应冻结协议和预测上下文均按任务归档。权重和协议的 SHA-256 已逐项与表 6 运行清单核对。

预测骨干为 models.py 中的 MultiTaskProcessModel。共享 Transformer 的隐藏维度为 128，有 2 层、4 个注意力头。活动、时间、静态和动态属性进入编码器，模式门控用于下一活动分支，另外两项主任务为下一活动发生时间和流程结果。trainer.py 实现分类、时间、辅助预测损失和不确定性加权。正式对比使用冻结模型。

## 按功能查找文件

| 目录 | 内容与入口 |
| --- | --- |
| 00_provenance | AutoDL 归档清单、运行迁移记录和校验结果。working_copy_comparison 单独保留当前工作区的 experiment_runner.py 供区分版本 |
| 01_prediction_training | models.py（预测网络）、trainer.py（训练和损失）、config.py（配置定义）、experiment_runner.py（原正式归档的实验入口）、constant.py |
| 02_data_preprocessing | data_loader.py、dataset.py、dataset_confs.py，以及输入编码和数据处理支持文件 |
| 03_patterns_search_rules | pattern_discovery/pattern_mining.py、pattern_to_ltlf.py、pattern_miner.py、trace_alignment.py、模式匹配工具和冻结模式缓存 |
| 04_counterfactual_search | cf_search/methods.py（正式严格搜索 strict_dfa_outcome）、cf_search/budget.py（预算）、cf_search/formal_bridge.py（冻结模型与搜索的连接）、counterfactual.py（DFA 管理和原有生成器），以及搜索支持文件 |
| 05_audit_evaluation | 独立 Audit 规则、audit_constraints.py、audit_rescore.py、cf_bridge_results.py、cf_query_metrics.py、配对统计和评估代码 |
| 06_experiment_and_baseline | run_cf_formal_bridge.py（正式对比入口）、实验配置、AutoDL 依赖、Buliga 对照实现及其许可、验收脚本 |
| 07_frozen_models_and_protocols | by_dataset 下存放七份权重及对应的 protocol.json、oracle_context.pickle、audit_policy.json，另有查询清单、冻结协议支持代码和正式运行依赖 |
| 08_paper_evidence | 论文表格对应的结果来源及汇总。表 3、表 4–5 和图 6–9、表 6 分开存放 |

## 结果与版本对应关系

- 表 3 为历史消融。英文稿的源文件注明它来自 2026-06-10 的 all_results/all_results.csv，指标版本为 v2。这批结果单独保留，正式对比的七份 seed 42 权重对应后续实验。
- 表 4–5 和图 6–9 的正式对比沿用 outputs/p6b_formal_autodl_v1 的保存结果。论文共同适用队列的核算可从“复核与汇总.json”和“论文实验表格.tex”追踪。独立验收涉及候选保存后的补充规则复核。
- 表 6 沿用同批保存候选，另用 analysis/complete_cf_multitask_cpu.py 对查询和候选获取三项预测。原运行选取 61 个查询、195 个目标类候选。归档内有其运行清单、查询汇总和候选配对。
- 论文正式运行使用的原始 AutoDL 包是 D:/MMOE_UWL_cf/outputs/p6b_autodl_bundle_20260918_v2.tar.gz。其内部清单哈希与服务器迁移记录相符。正式验收记录中的桥接代码与已归档副本匹配。工作区后续的 experiment_runner.py 与原包版本有差异，归档以原包版本为主。
- 功能目录含支持组件，保留的代码文件可能同时实现其他变体或历史路径。正式主方法以保存协议和 strict_dfa_outcome 为准。

## 校验与原目录结构

FILE_INVENTORY.csv 逐文件记录分类目录、原相对路径、来源归档及 SHA-256。00_provenance/verified_sources.json 保存检查点、冻结协议和通过验收的代码指纹。此整理包内的源码、冻结权重和协议均按原字节复制。

功能目录便于阅读和定位。Python 导入以及冻结协议的文件引用依赖原项目相对路径。需要重建原目录结构时，可执行下面的命令，它会校验哈希并复制到一个全新的目录。

    .\Restore-OriginalLayout.ps1 -Destination 'D:\CMT-CF_Restored'

整理包的范围是源码、权重、冻结配置和选定的结果证据。原始事件日志及完整的中间物理候选记录位于原项目。完整冻结依赖可从上文的原始 AutoDL 包获取。独立复现仍需准备依赖环境、原始日志和协议路径迁移。服务器正式对比的环境见 configs/autodl_requirements.txt 和 00_provenance/AUTODL_RUN.md。
