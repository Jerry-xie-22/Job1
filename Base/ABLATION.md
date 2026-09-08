# Base 核心模块消融实验

运行脚本会使用相同的数据、随机种子和超参数，依次执行完整模型和三组消融实验。
完整模型需要重新运行，因为本次修复了 MoE 分类器参数未进入优化器以及早停未保存
MoE 参数的问题，旧结果与修复后的消融结果不属于完全相同的训练条件。

| `ablation_mode` | 标签相似度 MoE 分类器 | `label_cons_loss` | 分类损失 |
|---|---:|---:|---|
| `full` | 保留 | 保留 | `CrossEntropyLoss(sim_scores, label_ids)` + `label_cons_loss` |
| `label_cons_only` | 关闭 | 保留 | `CrossEntropyLoss(logits, label_ids)` |
| `label_classifier_only` | 保留 | 关闭 | `CrossEntropyLoss(sim_scores, label_ids)` |
| `without_both` | 关闭 | 关闭 | `CrossEntropyLoss(logits, label_ids)` |

服务器运行：

```bash
cd /public/home/202420144954/job1/Base
sbatch examples/run_mag_bert_ablation.sh
```

四组汇总结果写入：

```text
results_ablation/components/component_ablation.csv
```

CSV 会记录 `ablation_mode`。每组实验的逐样本预测和逐类别指标使用带模式名的独立
TSV 文件，避免后运行的实验覆盖前一组结果。
