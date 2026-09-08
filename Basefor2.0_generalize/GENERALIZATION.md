# MIntRec2.0 → MIntRec：共同 20 类泛化实验

本目录以 `Basefor2.0` 为基础。实验使用 MIntRec2.0 的共同 20 类 train 训练、
共同 20 类 dev 验证并选择模型，最后在 MIntRec 的完整 test 上测试。

## 标签和 TSV

`data/__init__.py` 中 MIntRec2.0 的标签已缩减并重排为 MIntRec 的顺序：

```text
Complain, Praise, Apologise, Thank, Criticize, Agree, Taunt, Flaunt,
Joke, Oppose, Comfort, Care, Inform, Advise, Arrange, Introduce,
Leave, Prevent, Greet, Ask for help
```

仓库包含从原始 MIntRec2.0 TSV 按标签过滤得到的文件：

- `MIntRec2.0_train_20.tsv`：4,125 条，排除 2,040 条新增类别样本。
- `MIntRec2.0_dev_20.tsv`：726 条，排除 380 条新增类别样本。

两个文件保留原始表头、行顺序和全部列。代码默认直接读取它们；路径可用
`--train_tsv_path` 和 `--dev_tsv_path` 修改。MIntRec test 共 445 条，均属于共同
20 类，因此全部保留。标签、分类输出以及标签语义对比学习统一使用上述顺序。

## 服务器文件

数据目录仍用于读取测试 TSV 和特征：

```text
/public/home/202420144954/MIntRec-TCLMAP_generalize/MIntRec2.0/
  test.tsv                         # MIntRec 1.0 test
  audio_data/audio_feats.pkl       # MIntRec2.0 train/dev
  audio_data/audio_feats1.pkl      # MIntRec test
  video_data/video_feats.pkl       # MIntRec2.0 train/dev，256 维
  video_data/video_feats1.pkl      # MIntRec test，256 维
```

每种模态的两个 PKL 必须来自兼容的特征提取器并具有相同宽度。代码会检查
train/dev/test 的宽度；不一致时立即报错，不再通过补零强行兼容。时间长度统一为
文本 76、视频 230、音频 480；短序列按配置补齐，长序列保留前面的帧。

分类和标签语义对比学习使用 MIntRec 的 20 类标签描述文件：

```text
/public/home/202420144954/job2/Base_for_emo_mintrec10_c2f_inject_large/data/label_descriptions_mintrec.pt
```

路径可用 `--label_descriptions_path` 修改。该文件中的数字标签顺序必须与
`data/__init__.py` 的 20 类顺序一致。

## 运行

```bash
cd /public/home/202420144954/job1/Basefor2.0_generalize
sbatch examples/run_generalize.sh
```

结果写入 `results_generalize/`，每轮还会生成数据报告，记录样本数、标签顺序、
特征路径、宽度和截断数量。超参数只应根据过滤后的 2.0 dev 选择。

本地轻量测试：

```bash
python -m unittest discover -s Basefor2.0_generalize/tests -v
```

完整训练需要服务器的 PKL、BERT、标签描述文件和 PyTorch/Transformers 环境。
