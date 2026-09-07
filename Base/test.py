import torch
import torch.nn as nn

# 定义LSTM层
pred_output_position_inclu_blank = nn.LSTM(768, 31, num_layers=2, batch_first=True)

# 构造输入，形状为 (batch_size=16, seq_len=480, input_size=768)
input_tensor = torch.randn(16, 480, 768)  # 使用torch.randn生成随机数据

# 传递到LSTM中
output, (h_n, c_n) = pred_output_position_inclu_blank(input_tensor)

# 输出结果
print(output.shape)  # 应该是 (16, 480, 31)

print(output)