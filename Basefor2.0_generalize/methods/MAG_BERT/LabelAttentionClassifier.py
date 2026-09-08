import torch
import torch.nn as nn
import torch.nn.functional as F


class LabelAttentionClassifier(nn.Module):
    """
    改进版：单头 + 语义一致注意力 + 可学习温度
    cls_output:   [B, D]
    label_embeds: [L, K, D]  # L labels, each with K descriptions
    returns:
        logits:      [B, L]
        att_weights: [B, L, K]
    """

    def __init__(self, hidden_dim, dropout=0.1, learnable_temp=True):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        if learnable_temp:
            self.logit_scale = nn.Parameter(torch.ones(()) * torch.log(torch.tensor(10.0)))
        else:
            self.logit_scale = 1.0

    def forward(self, cls_output, label_embeds):
        B, D = cls_output.shape
        L, K, D2 = label_embeds.shape
        assert D == D2

        # 1. L2 normalize (critical for cosine similarity)
        cls_norm = F.normalize(cls_output, dim=-1)           # [B, D]
        label_norm = F.normalize(label_embeds, dim=-1)       # [L, K, D]

        # 2. Compute raw cosine similarities: [B, L, K]
        sim_raw = torch.einsum('bd, lkd -> blk', cls_norm, label_norm)

        # 3. Attention weights over K descriptions (per label)
        #    Use sim_raw itself as attention logits!
        att_logits = sim_raw * self.logit_scale              # [B, L, K]
        att_weights = F.softmax(att_logits, dim=-1)          # [B, L, K]
        att_weights = self.dropout(att_weights)

        # 4. Weighted sum of similarities → final logit per label
        logits = (att_weights * sim_raw).sum(dim=-1)         # [B, L]

        return logits, att_weights





# 实验发现，最终Moe的专家数为0时效果最好，此时退化为基于相似度的分类器
class MoELabelAttentionClassifier(nn.Module):
    def __init__(self, hidden_dim, num_experts=6, dropout=0.1):
        super().__init__()
        self.num_experts = num_experts
        self.dropout = nn.Dropout(dropout)
        if num_experts > 0:
            self.logit_scales = nn.Parameter(torch.ones(num_experts) * 2.3026)
            self.router = nn.Linear(hidden_dim, num_experts)

        # 每个 expert 有自己的可学习 scale（替代统一 logit_scale）
        self.logit_scales = nn.Parameter(torch.ones(num_experts) * 2.3026)  # init to log(10)

        # 路由器：根据 cls 向量决定信任哪个 expert
        self.router = nn.Linear(hidden_dim, num_experts)

    def forward(self, cls_output, label_embeds):
        B, D = cls_output.shape
        L, K, D2 = label_embeds.shape
        assert D == D2

        cls_norm = F.normalize(cls_output, dim=-1)      # [B, D]
        label_norm = F.normalize(label_embeds, dim=-1)  # [L, K, D]

        sim_raw = torch.einsum('bd, lkd -> blk', cls_norm, label_norm)  # [B, L, K]

        # --- MoE: 每个 expert 计算自己的注意力和 logits ---
        expert_logits_list = []
        for i in range(self.num_experts):
            scale = self.logit_scales[i].exp().clamp(max=100.0)
            att_logits = sim_raw * scale                     # [B, L, K]
            att_weights = F.softmax(att_logits, dim=-1)
            att_weights = self.dropout(att_weights)
            expert_logit = (att_weights * sim_raw).sum(dim=-1)  # [B, L]
            expert_logits_list.append(expert_logit.unsqueeze(-1))  # [B, L, 1]
        if(len(expert_logits_list) == 0):
            # No experts, use zero logits
            return sim_raw.sum(dim=-1), None
        # Stack: [B, L, num_experts]
        all_expert_logits = torch.cat(expert_logits_list, dim=-1)

        # --- Router: decide which expert to trust per sample ---
        router_logits = self.router(cls_output)          # [B, num_experts]
        router_weights = F.softmax(router_logits, dim=-1)  # [B, num_experts]

        # Expand router weights to [B, 1, num_experts] for broadcasting
        router_weights = router_weights.unsqueeze(1)     # [B, 1, E]

        # Weighted sum over experts
        final_logits = (all_expert_logits * router_weights).sum(dim=-1)  # [B, L]

        return final_logits, router_weights.squeeze(1)  # 可选：返回路由权重用于分析




# class DegeneratedMoELabelClassifier(nn.Module):
#     def __init__(self, hidden_dim, num_experts=5, dropout=0.1):
#         super().__init__()
#         self.num_experts = num_experts
#         self.dropout = nn.Dropout(dropout)

#         # 固定 scale = 1（即不放缩）
#         self.register_buffer("logit_scales", torch.zeros(num_experts))

#         # Router 仍然存在，但输出固定平均
#         self.router = nn.Linear(hidden_dim, num_experts)

#         # 冻结 router 权重 (让它永远产生 0 → softmax = uniform)
#         for p in self.router.parameters():
#             p.requires_grad = False
#             nn.init.constant_(p, 0.0)

#     def forward(self, cls_output, label_embeds):
#         B, D = cls_output.shape
#         L, K, D2 = label_embeds.shape
#         assert D == D2

#         cls_norm = F.normalize(cls_output, dim=-1)
#         label_norm = F.normalize(label_embeds, dim=-1)

#         sim = torch.einsum('bd, lcd -> blc', cls_norm, label_norm) # 相似度计算: [B, num_labels, 3]
#         sim_scores = sim.sum(dim=-1)  # 每个标签的三个描述向量相似度求和 [B, num_labels]

#         # # baseline sim_raw
#         # sim_raw = torch.einsum("bd,lkd->blk", cls_norm, label_norm)  # [B, L, K]

#         # # --- 退化 attention：softmax 后 = 1/K 的均匀分布 ---
#         # att_weights = torch.ones(B, L, K, device=sim_raw.device) / K
#         # att_weights = self.dropout(att_weights)

#         # # --- 每个 expert 输出一样的 baseline logits ---
#         # expert_logit = (att_weights * sim_raw).sum(dim=-1)  # [B, L]
#         # expert_logit = expert_logit.unsqueeze(-1).repeat(1, 1, self.num_experts)

#         # # --- Router 退化：softmax = 1/E 平均 ---
#         # router_logits = self.router(cls_output)      # 全 0
#         # router_weights = F.softmax(router_logits, dim=-1)  # uniform: [B, E]

#         # router_weights = router_weights.unsqueeze(1)  # [B, 1, E]

#         # # --- 最终融合：其实就是 baseline ---
#         # final_logits = (expert_logit * router_weights).sum(dim=-1)  # [B, L]

#         return sim_scores





