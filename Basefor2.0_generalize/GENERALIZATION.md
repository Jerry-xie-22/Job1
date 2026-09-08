# MIntRec2.0 → MIntRec

以 Basefor2.0 为基础的独立副本。2.0 train 训练、2.0 dev 验证并选择模型，
最后在 1.0 test 上测试。保留 2.0 全部 30 类及原标签编号、标签描述文件和模型方法。
1.0 的 20 类全部属于这 30 类；全部测试样本保留，标签通过名称映射到 2.0 编号。
预测仍在 30 类中选择，预测为仅 2.0 存在的类别也计为错误，不进行 20 类掩码。
指标沿用 Basefor2.0：accuracy 和 sklearn 默认宏/加权指标；宏平均类别集合为
真实标签与预测标签的并集，不是强制固定为 20 类。

服务器布局（注意大小写）：

```text
/public/home/202420144954/MIntRec-TCLMAP/MIntRec2.0/
  train.tsv                       # 原始 2.0 train
  dev.tsv                         # 原始 2.0 dev
  test.tsv                        # 原始 1.0 test，保留表头
  audio_data/audio_feats.pkl      # 2.0 train/dev
  audio_data/audio_feats1.pkl     # 1.0 test
  video_data/video_feats.pkl      # 2.0 train/dev
  video_data/video_feats1.pkl     # 1.0 test
  label_descriptions_mintrec2.0.pt # 原模型需要的 30 类描述
```

2.0 TSV 使用 Dialogue_id / Utterance_id / Text / Label，特征键为 diaN_uttM；
1.0 TSV 使用 season / episode / clip / text / label，特征键为 season_episode_clip。
所有模态使用相同的解析样本列表；未知标签、缺失特征、错误格式明确报错。
文件名可通过 --audio_feats_path、--test_audio_feats_path 等参数指定。

序列上限取本目录 benchmarks 中两个数据集的较大值：文本 76、视频 230、音频 480。
音视频支持 (T,D) / (T,1,D)，按实际源/目标宽度最大值统一，较窄特征右侧补零，
输出 float32；时间维沿用 padding_mode / padding_loc，超长保留前面的帧。
数据报告记录特征路径、原始宽度、统一尺寸、截断数量和每个 split 的覆盖率。
宽度补零只实现形状兼容，不能保证不同特征提取器的语义空间一致。

运行（使用服务器现有 mvcl-daf 环境、bert-large-uncased 路径）：

```bash
cd /public/home/202420144954/job1/Basefor2.0_generalize
sbatch examples/run_generalize.sh
```

启动脚本保留原 mag_bert 超参数网格，seed=4；目前 3 个 dropout × 6 个 lr ×
9 个 num_experts，共 162 次训练。单次实验请先将 configs/mag_bert.py 中的列表
改为在 2.0 验证集上选定的单元素值，保留 --tune 用于展开列表。
不要依据 1.0 测试结果挑选超参数。新输出在 results_generalize/、logs/generalize/，
每轮写 *_data_report.json。复制来的历史 results 文件不是本次实验结果。
其他旧 examples 脚本仍为原实验路径，本实验使用 run_generalize.sh。

轻量回归测试（Python + NumPy；集成测试替代了 PyTorch/Tokenizer 边界）：

```bash
python -m unittest discover -s Basefor2.0_generalize/tests -v
```

真实 TSV 样本测试使用本地 MInteRec_data；未提供该目录时跳过该测试。
完整模型训练需服务器 PKL、BERT、标签描述和 PyTorch/Transformers 环境。
