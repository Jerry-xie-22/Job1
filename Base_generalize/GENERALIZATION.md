# MIntRec → MIntRec2.0 泛化实验

本目录是 Base 的独立副本。本阶段只实现 MIntRec 的 train/dev 训练和选择模型，
随后在 MIntRec2.0 的 test 上评估。仅保留 MIntRec 已有的 20 类，标签编号沿用 Base；
其余标签（包括 UNK）排除。结果不是完整 MIntRec2.0 测试集的 30 类或开放集指标。

服务器数据布局（目录名大小写必须为 MIntRec）：

```text
/public/home/202420144954/MIntRec-TCLMAP/MIntRec/
  train.tsv                 # MIntRec train
  dev.tsv                   # MIntRec dev
  test.tsv                  # MIntRec2.0 test，保留原始表头
  audio_data/audio_feats.pkl
  audio_data/audio_feats2.pkl
  video_data/video_feats.pkl
  video_data/video_feats2.pkl
```

1.0 使用 season_episode_clip ID、text/label 列；2.0 使用 dia{Dialogue_id}_utt{Utterance_id}
ID、Text/Label 列。文本、标签、音视频使用同一份筛选后的样本列表。缺失特征、
错误表头、非有限值、空特征或不一致的特征宽度会明确报错。

统一序列长度为文本 50、视频 230、音频 480（两个已有基准配置的较大值）；
超长序列从头截断，短序列遵循原 padding_mode/padding_loc。
音视频支持 (T,D) 和 (T,1,D)，按实际源/目标宽度的最大值补零到相同宽度，
使用 float32。模型创建前更新 args 中的特征维度。宽度补零只解决形状兼容，
不保证不同特征提取器的语义空间一致；严谨对比最好使用相同提取器。

使用原 mvcl-daf 环境和服务器 BERT、标签描述向量路径。执行：

```bash
cd /public/home/202420144954/job1/Base_generalize
sbatch examples/run_generalize.sh
```

新脚本使用独立的 results_generalize、logs/generalize 和 outputs 目录。
每次运行会写入 *_data_report.json，记录标签顺序、保留比例、排除标签及数量、
特征文件路径、原始宽度、统一尺寸和截断样本数。
验证集只来自 1.0；不应按目标测试集分数选择超参数。

测试特征文件名可用 --test_audio_feats_path / --test_video_feats_path 修改。
训练参数沿用 Base 的 mag_bert 配置，--tune 用于展开其中的单元素列表。
复制得到的其他旧 examples 脚本仍指向原实验；本实验使用 run_generalize.sh。
复制得到的历史 results 文件不代表本次泛化实验结果。

本地轻量测试（Python + NumPy，不需要 CUDA 或下载 BERT）：

```bash
python -m unittest discover -s Base_generalize/tests -v
```

完整训练仍需服务器中的特征 PKL、预训练模型和 PyTorch/Transformers 环境。
