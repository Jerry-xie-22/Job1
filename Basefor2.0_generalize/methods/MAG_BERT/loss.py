
from torch import nn
import torch
import torch.nn.functional as F


class LabelAttentionPooling(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.W_q = nn.Linear(hidden_dim, hidden_dim)
        self.W_k = nn.Linear(hidden_dim, hidden_dim)
        self.W_v = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, h, label_descs):
        """
        h: [batch, d] 多模态融合后的输入
        label_descs: [batch, 3, d] 每个标签的三个描述向量
        """
        q = self.W_q(h).unsqueeze(1)        # [batch, 1, d]
        k = self.W_k(label_descs)           # [batch, 3, d]
        v = self.W_v(label_descs)           # [batch, 3, d]

        # 注意力权重
        attn_scores = torch.bmm(q, k.transpose(1, 2)) / (h.size(-1) ** 0.5)  # [batch, 1, 3]
        attn_weights = F.softmax(attn_scores, dim=-1)                        # [batch, 1, 3]

        # 动态组合后的标签向量
        d_star = torch.bmm(attn_weights, v).squeeze(1)  # [batch, d]

        return d_star






# class TripletContrastiveLoss(nn.Module):
#     def __init__(self, device, temperature=0.07):
#         super(TripletContrastiveLoss, self).__init__()
#         self.temperature = temperature
#         self.loss_scale = nn.Parameter(torch.ones(1).to(device))
#         self.device = device
        

#     def forward(self, cls_output, label_embed):
#         """
#         Args:
#             cls_output: [B, D] -
#             label_embed: [B, 3, L, D] - 每个样本的三个标签描述表示，L其实就是1
#         Returns:
#             loss: scalar
#         """
#         # self.loss_scale.to(self.device)
#         B, D = cls_output.shape
        
#         # 模型的输出
#         anchor = F.normalize(cls_output)           # [B, D]
#         # 每个样本对应的标签有三个描述，所以有三个768维的向量。
#         positive = F.normalize(label_embed.view(B, 3, -1), dim=-1)       # [B, 3, D]
#         # print(f"  anchor shape: {anchor.shape}")
#         # print(f"  positive shape: {positive.shape}")

#         # 2. reshape正样本：每个 anchor 配 3 个正样本
#         anchor = anchor.unsqueeze(1).expand(-1, 3, -1)     # [B, 3, D]
#         # print(f"  anchor shape: {anchor.shape}")
#         # print(f"  B: {B}")
#         # print(f"  L: {L}")
#         # print(f"  D: {D}")
#         anchor_flat = anchor.reshape(B * 3, D)
#         # print(f"  anchor_flat shape: {anchor_flat.shape}")
#         positive_flat = positive.reshape(B * 3, D)
#         # print(f"  positive_flat shape: {positive_flat.shape}")


#         # 3. 构建对比矩阵（正样本 vs 全部正样本 + 负样本）
#         # 所有正样本作为对比库
#         all_positives = positive_flat.detach()             # [B*3, D]
#         logits = torch.matmul(anchor_flat, all_positives.T) / self.temperature  # [B*3, B*3]
#         # print(f"  logits shape: {logits.shape}")

#         # 4. 构建标签（同一个样本内部的3个正样本是匹配的，其他为负）
#         labels = torch.arange(B).repeat_interleave(3).to(cls_output.device)  # [B*3]
#         mask = torch.eq(labels.unsqueeze(1), labels.unsqueeze(0)).float()      # [B*3, B*3]
#         logits_mask = 1 - torch.eye(B*3, device=cls_output.device)
#         mask = mask * logits_mask

#         # 5. InfoNCE loss
#         logits_max, _ = torch.max(logits, dim=1, keepdim=True)
#         logits = logits - logits_max.detach()

#         exp_logits = torch.exp(logits) * logits_mask
#         log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-12)

#         mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1) + 1e-12)
#         loss = -mean_log_prob_pos.mean()
#         scaled_loss = self.loss_scale * loss
#         return scaled_loss



# Anchor = 当前样本的多模态融合输出 [𝐵,𝐷]
# 正样本 = 当前样本真实标签对应的 3 个描述向量
# 负样本 = 其它标签的描述向量

class TripletContrastiveLoss(nn.Module):
    def __init__(self, device, temperature=0.07):
        super(TripletContrastiveLoss, self).__init__()
        self.temperature = temperature
        self.loss_scale = nn.Parameter(torch.ones(1).to(device))
        self.device = device
        

    def forward(self, cls_output, label_embed, label_ids):
        """
        Args:
            cls_output: [B, D] -
            label_embed: [B, 3, L, D] - 每个样本的三个标签描述表示，L其实就是1
            all_label_embed: [20, 3, L, D] - 所有标签的描述
        Returns:
            loss: scalar
        """


        loss = label_contrastive_loss_multi_pos(cls_output, label_embed, label_ids)

        # # self.loss_scale.to(self.device)
        # B, D = cls_output.shape
        
        # # 模型的输出
        # anchor = F.normalize(cls_output)           # [B, D]
        # # 每个样本对应的标签有三个描述，所以有三个768维的向量。
        # positive = F.normalize(label_embed.view(B, 3, -1), dim=-1)       # [B, 3, D]
        # # print(f"  anchor shape: {anchor.shape}")
        # # print(f"  positive shape: {positive.shape}")

        # # 2. reshape正样本：每个 anchor 配 3 个正样本
        # anchor = anchor.unsqueeze(1).expand(-1, 3, -1)     # [B, 3, D]
        # # print(f"  anchor shape: {anchor.shape}")
        # # print(f"  B: {B}")
        # # print(f"  L: {L}")
        # # print(f"  D: {D}")
        # anchor_flat = anchor.reshape(B * 3, D)
        # # print(f"  anchor_flat shape: {anchor_flat.shape}")
        # positive_flat = positive.reshape(B * 3, D)
        # # print(f"  positive_flat shape: {positive_flat.shape}")


        # # 3. 构建对比矩阵（正样本 vs 全部正样本 + 负样本）
        # # 所有正样本作为对比库
        # all_positives = positive_flat.detach()             # [B*3, D]
        # logits = torch.matmul(anchor_flat, all_positives.T) / self.temperature  # [B*3, B*3]
        # # print(f"  logits shape: {logits.shape}")

        # # 4. 构建标签（同一个样本内部的3个正样本是匹配的，其他为负）
        # labels = torch.arange(B).repeat_interleave(3).to(cls_output.device)  # [B*3]
        # mask = torch.eq(labels.unsqueeze(1), labels.unsqueeze(0)).float()      # [B*3, B*3]
        # logits_mask = 1 - torch.eye(B*3, device=cls_output.device)
        # mask = mask * logits_mask

        # # 5. InfoNCE loss
        # logits_max, _ = torch.max(logits, dim=1, keepdim=True)
        # logits = logits - logits_max.detach()

        # exp_logits = torch.exp(logits) * logits_mask
        # log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-12)

        # mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1) + 1e-12)
        # loss = -mean_log_prob_pos.mean()
        scaled_loss = self.loss_scale * loss
        return scaled_loss



class SingleContrastiveLoss(nn.Module):
    def __init__(self, device, temperature=0.07):
        super(SingleContrastiveLoss, self).__init__()
        self.temperature = temperature
        self.loss_scale = nn.Parameter(torch.ones(1).to(device))
        self.device = device
        

    def forward(self, cls_output, label_embed, label_ids):
        """
        Args:
            cls_output: [B, D] -
            label_embed: [B, 3, L, D] - 每个样本的三个标签描述表示，L其实就是1
            all_label_embed: [20, 3, L, D] - 所有标签的描述
        Returns:
            loss: scalar
        """


        loss = label_contrastive_loss_single(cls_output, label_embed, label_ids)
        scaled_loss = self.loss_scale * loss
        return scaled_loss


class KContrastiveLoss(nn.Module):
    def __init__(self, device, temperature=0.07):
        super(KContrastiveLoss, self).__init__()
        self.temperature = temperature
        self.loss_scale = nn.Parameter(torch.ones(1).to(device))
        self.device = device
        

    def forward(self, cls_output, label_embed, label_ids):
        """
        Args:
        cls_output: [B, D]
        label_desc: [T, k, D]  每个标签 k 个描述 embedding
        labels: [B]

        Returns:
            loss: scalar
        """


        loss = label_contrastive_loss_K_pos(cls_output, label_embed, label_ids)
        scaled_loss = self.loss_scale * loss
        return scaled_loss




def label_contrastive_loss_multi_pos(cls_output, label_desc, labels, temperature=0.07):
    """
    多正样本 InfoNCE (方案 a)

    Args:
        cls_output: [B, D]  模型输出的句子级向量 (anchor)
        label_desc: [T, 3, D]  所有标签的 3 个描述向量 (已编码好)
        labels: [B]  每个样本的真实标签 id (0~T-1)
        temperature: 温度参数
    Returns:
        loss: scalar
    """
    B, D = cls_output.shape
    T = label_desc.size(0)

    # 归一化
    cls_output = F.normalize(cls_output, dim=-1)   # [B, D]
    label_desc = F.normalize(label_desc, dim=-1)   # [T, 3, D]
    
    # print(cls_output.shape)
    # print(label_desc.shape)
    # 计算相似度: [B, T, 3]，输出与每一个标签三个描述的相似度
    logits = torch.einsum("bd,tjd->btj", cls_output, label_desc) / temperature

    # 展开到 [B, T*3]
    logits = logits.view(B, T * 3)

    # 构造正样本 mask: [B, T*3]
    mask = torch.zeros(B, T * 3, device=cls_output.device)
    for i, y in enumerate(labels):
        mask[i, y*3:(y+1)*3] = 1

    # denominator: 所有样本的 exp 相似度和
    exp_logits = torch.exp(logits)   # [B, T*3]
    denominator = exp_logits.sum(dim=1, keepdim=True)  # [B, 1]

    # numerator: 从exp_logits找出正样本的相似度，把负样本的相似度变为0
    numerator = exp_logits * mask  # [B, T*3]

    # 拿numerator除以denominator，分子包含了负样本
    log_prob_pos = torch.log(numerator / (denominator + 1e-12) + 1e-12)  # [B, T*3]

    # 去除分子为负样本的部分，只保留正样本的损失。
    loss = -(log_prob_pos * mask).sum(dim=1) / (mask.sum(dim=1) + 1e-12)  # [B]
    loss = loss.mean()

    return loss





def label_contrastive_loss_single(
        cls_output, label_desc, labels, temperature=0.07):
    """
    单正样本 InfoNCE（每个标签只有一个描述 embedding）

    Args:
        cls_output: [B, D]
        label_desc: [T, 1, D] 或 [T, D]
        labels: [B]
    """
    B, D = cls_output.shape

    # 如果 label_desc 是 [T, 1, D]，压掉中间维度
    if label_desc.dim() == 3:
        label_desc = label_desc.squeeze(1)  # -> [T, D]

    T = label_desc.size(0)

    # 归一化
    cls_output = F.normalize(cls_output, dim=-1)
    label_desc = F.normalize(label_desc, dim=-1)

    # 相似度 logits: [B, T]
    logits = torch.einsum("bd,td->bt", cls_output, label_desc) / temperature

    # labels: [B]，cross entropy 的 target
    loss = F.cross_entropy(logits, labels)

    return loss



def label_contrastive_loss_K_pos(cls_output, label_desc, labels, temperature=0.07):
    """
    多正样本 InfoNCE，支持 label_desc=[T, k, D]，k>=1

    Args:
        cls_output: [B, D]
        label_desc: [T, k, D]  每个标签 k 个描述 embedding
        labels: [B]
    """
    B, D = cls_output.shape
    T, K, _ = label_desc.shape   # 现在 K=2

    # 归一化
    cls_output = F.normalize(cls_output, dim=-1)   
    label_desc = F.normalize(label_desc, dim=-1)

    # 相似度: [B, T, K]
    logits = torch.einsum("bd,tkd->btk", cls_output, label_desc) / temperature

    # 展平 [B, T*K]
    logits = logits.view(B, T * K)

    # 构造正样本 mask: [B, T*K]
    mask = torch.zeros(B, T * K, device=cls_output.device)
    for i, y in enumerate(labels):
        mask[i, y*K : (y+1)*K] = 1   # 现在 K=2，对应 2 个正样本

    # exp
    exp_logits = torch.exp(logits)
    denom = exp_logits.sum(dim=1, keepdim=True)

    # numerator
    numerator = exp_logits * mask  # 正样本保留，负样本归零

    # log probability of positive samples
    log_prob_pos = torch.log(numerator / (denom + 1e-12) + 1e-12)

    # 取正样本损失
    loss = -(log_prob_pos * mask).sum(dim=1) / (mask.sum(dim=1) + 1e-12)

    return loss.mean()
