import torch.nn.functional as F
import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import CrossEntropyLoss, MSELoss
from transformers import BertPreTrainedModel
from transformers.models.bert.modeling_bert import BertEmbeddings, BertEncoder, BertPooler
from ..SubNets.AlignNets import AlignSubNet

class MAG(nn.Module):
    def __init__(self,  config, args):
        super(MAG, self).__init__()
        self.args = args

        # 特征对齐网络
        if self.args.need_aligned:
            self.alignNet = AlignSubNet(args, args.aligned_method)

        # 获取输入的特征维度
        text_feat_dim, audio_feat_dim, video_feat_dim = args.text_feat_dim, args.audio_feat_dim, args.video_feat_dim
        # 定义一个全连接层，输入视频和文本特征的连接结果作为x，输出y的维度为文本的维度
        self.W_hv = nn.Linear(video_feat_dim + text_feat_dim, text_feat_dim)
        self.W_ha = nn.Linear(audio_feat_dim + text_feat_dim, text_feat_dim)
        # 处理视频和音频的特征，变换为文本的维度
        self.W_v = nn.Linear(video_feat_dim, text_feat_dim)
        self.W_a = nn.Linear(audio_feat_dim, text_feat_dim)
        # 控制了最终融合特征的加权比例
        self.beta_shift = args.beta_shift
        # 归一化处理
        self.LayerNorm = nn.LayerNorm(config.hidden_size)
        # 进行dropout操作
        self.dropout = nn.Dropout(args.dropout_prob)

    def forward(self, text_embedding, visual, acoustic):
        # 用于数值稳定性确保
        eps = 1e-6
        # 对齐
        if self.args.need_aligned:
            text_embedding, acoustic, visual  = self.alignNet(text_embedding, acoustic, visual)
        
        # 计算视频和音频特征的权重
        weight_v = F.relu(self.W_hv(torch.cat((visual, text_embedding), dim=-1)))
        weight_a = F.relu(self.W_ha(torch.cat((acoustic, text_embedding), dim=-1)))
        
        # 计算视频和音频特征的加权和（视频和音频融合）
        h_m = weight_v * self.W_v(visual) + weight_a * self.W_a(acoustic)

        # 计算文本特征和融合特征（视频+文本）的 L2 范数，L2 范数被用来衡量特征向量的大小（即模长）
        em_norm = text_embedding.norm(2, dim=-1)
        hm_norm = h_m.norm(2, dim=-1)

        # 计算时如果 hm_norm 为零，可能会导致数值不稳定（比如除零错误）
        hm_norm_ones = torch.ones(hm_norm.shape, requires_grad=True).to(self.args.device)
        # 如果 hm_norm 为零，则将其替换为 1，确保后续计算不出错。
        # TODO:直接替换为1不会影响原先的效果吗？比如下面的比例计算，换为1，比例差别会很大
        hm_norm = torch.where(hm_norm == 0, hm_norm_ones, hm_norm)
        
        # 通过比较 em_norm 和 hm_norm 的比例，计算一个加权因子 thresh_hold。
        # 这个比例反映了文本特征与融合特征的相对“大小”，并通过 beta_shift 进行调节。
        # 为什么加eps呢，防止hm_norm中接近0的数值导致除了以后数值很大，导致数值不稳定
        thresh_hold = (em_norm / (hm_norm + eps)) * self.beta_shift

        ones = torch.ones(thresh_hold.shape, requires_grad=True).to(self.args.device)

        # 通过 torch.min 函数，确保 thresh_hold 的值不会超过 1。因为比例不可能超过1
        alpha = torch.min(thresh_hold, ones)
        alpha = alpha.unsqueeze(dim=-1)

        # alpha 被用来加权融合特征 h_m，确保文本特征与融合特征的加权比例合适，相当于控制融合比例
        acoustic_vis_embedding = alpha * h_m

        # 加权后的特征与文本特征加和，并通过 LayerNorm 和 Dropout
        embedding_output = self.dropout(
            self.LayerNorm(acoustic_vis_embedding + text_embedding)
        )
        # 融合结果的维度跟text的维度是一样的，因为要把融合结果输入到bert模型中
        return embedding_output














# 继承自 BertPreTrainedModel
class MAG_BertModel(BertPreTrainedModel):
    def __init__(self, config, args):
        super().__init__(config)
        self.config = config
        # 标准的 BERT embedding 层（包含 word_embeddings, position_embeddings, token_type_embeddings，后接 LayerNorm + dropout）
        self.embeddings = BertEmbeddings(config)
        # BERT 的一组 Transformer 层，主要处理输入的嵌入数据，进行自注意力（Self-Attention）计算。
        self.encoder = BertEncoder(config)
        # BERT 模型中的池化层，取序列第一个 token （CLS）的 hidden-state，经过 Linear + tan
        self.pooler = BertPooler(config)
        # 融合模块
        # self.MAG = MAG(
        #     config, args
        # )

        self.MAG = MAG(
            config, args
        )
        # 此处有修改
        self.init_weights()

    # 下面这三个函数都是从bert模型里面复制过来的
    # 用于获取和设置 BERT 模型的输入嵌入层（word_embeddings）
    def get_input_embeddings(self):
        return self.embeddings.word_embeddings
    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value
    # 修剪注意力头，它会根据传入的 heads_to_prune 字典，修剪掉 BERT 模型中某些层的部分注意力头
    def _prune_heads(self, heads_to_prune):
        """ Prunes heads of the model.
            heads_to_prune: dict of {layer_num: list of heads to prune in this layer}
            See base class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)



    def forward(
        self,
        input_ids, # 文本的token ids
        visual, # 视频特征
        acoustic, # 音频特征
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        output_attentions=None,
        output_hidden_states=None,
    ):
    
        r"""
    Return:
        :obj:`tuple(torch.FloatTensor)` comprising various elements depending on the configuration (:class:`~transformers.BertConfig`) and inputs:
        last_hidden_state (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the model.
        pooler_output (:obj:`torch.FloatTensor`: of shape :obj:`(batch_size, hidden_size)`):
            Last layer hidden-state of the first token of the sequence (classification token)
            further processed by a Linear layer and a Tanh activation function. The Linear
            layer weights are trained from the next sentence prediction (classification)
            objective during pre-training.

            This output is usually *not* a good summary
            of the semantic content of the input, you're often better with averaging or pooling
            the sequence of hidden-states for the whole input sequence.
        hidden_states (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_hidden_states=True`` is passed or when ``config.output_hidden_states=True``):
            Tuple of :obj:`torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer)
            of shape :obj:`(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_attentions=True`` is passed or when ``config.output_attentions=True``):
            Tuple of :obj:`torch.FloatTensor` (one for each layer) of shape
            :obj:`(batch_size, num_heads, sequence_length, sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
        """
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the same time"
            )
        elif input_ids is not None:
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError(
                "You have to specify either input_ids or inputs_embeds")

        device = input_ids.device if input_ids is not None else inputs_embeds.device
        if attention_mask is None:
            attention_mask = torch.ones(input_shape, device=device)
        if token_type_ids is None:
            token_type_ids = torch.zeros(
                input_shape, dtype=torch.long, device=device)

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(
            attention_mask, input_shape, device
        )

        # If a 2D ou 3D attention mask is provided for the cross-attention
        # we need to make broadcastabe to [batch_size, num_heads, seq_length, seq_length]
        if self.config.is_decoder and encoder_hidden_states is not None:
            (
                encoder_batch_size,
                encoder_sequence_length,
                _,
            ) = encoder_hidden_states.size()
            encoder_hidden_shape = (
                encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(
                    encoder_hidden_shape, device=device)
            encoder_extended_attention_mask = self.invert_attention_mask(
                encoder_attention_mask
            )
        else:
            encoder_extended_attention_mask = None

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = self.get_head_mask(
            head_mask, self.config.num_hidden_layers)

        embedding_output = self.embeddings(
            input_ids=input_ids,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
            inputs_embeds=inputs_embeds,
        )




        # 特征融合结果，把bert的输入从text改为融合后的特征
        fused_embedding = self.MAG(embedding_output, visual, acoustic)

        





        # 融合以后的特征放到bert里面。
        encoder_outputs = self.encoder(
            fused_embedding,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_extended_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )
        # bert最后的所有输出
        sequence_output = encoder_outputs[0] # (B, L, D)
        # cls处的向量输出
        cls_output = sequence_output[:, 0, :]  # (B, D)
        pooled_output = self.pooler(sequence_output) 

        outputs = (sequence_output, pooled_output, cls_output) + encoder_outputs[
            1:
        ]  # add hidden_states and attentions if they are here
        # sequence_output, pooled_output, (hidden_states), (attentions)
        return outputs
    






class MAG_BertForSequenceClassification(BertPreTrainedModel):
    def __init__(self, config, args):
        super().__init__(config)
        self.num_labels = args.num_labels

        self.bert = MAG_BertModel(config, args)
        # 分类头
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, args.num_labels)

        self.init_weights()

    def forward(
        self,
        text,
        visual,
        acoustic,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`, defaults to :obj:`None`):
            Labels for computing the sequence classification/regression loss.
            Indices should be in :obj:`[0, ..., config.num_labels - 1]`.
            If :obj:`config.num_labels == 1` a regression loss is computed (Mean-Square loss),
            If :obj:`config.num_labels > 1` a classification loss is computed (Cross-Entropy).
    Returns:
        :obj:`tuple(torch.FloatTensor)` comprising various elements depending on the configuration (:class:`~transformers.BertConfig`) and inputs:
        loss (:obj:`torch.FloatTensor` of shape :obj:`(1,)`, `optional`, returned when :obj:`label` is provided):
            Classification (or regression if config.num_labels==1) loss.
        logits (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, config.num_labels)`):
            Classification (or regression if config.num_labels==1) scores (before SoftMax).
        hidden_states (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_hidden_states=True`` is passed or when ``config.output_hidden_states=True``):
            Tuple of :obj:`torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer)
            of shape :obj:`(batch_size, sequence_length, hidden_size)`.
            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_attentions=True`` is passed or when ``config.output_attentions=True``):
            Tuple of :obj:`torch.FloatTensor` (one for each layer) of shape
            :obj:`(batch_size, num_heads, sequence_length, sequence_length)`.
            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
        """
        input_ids, attention_mask, token_type_ids = text[:, 0], text[:, 1], text[:, 2]

        #前面outputs = (sequence_output, pooled_output, cls_output) + encoder_outputs[1:]
        outputs = self.bert(
            input_ids,
            visual,
            acoustic,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )

        # outputs[1]为pooled_output
        pooled_output = outputs[1]
        cls_output = outputs[2]

        # 分类
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        outputs = (logits,cls_output) + outputs[
            2:
        ]  # add hidden states and attention if they are here

        if labels is not None:
            if self.num_labels == 1:
                #  We are doing regression
                loss_fct = MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(
                    logits.view(-1, self.num_labels), labels.view(-1))
            outputs = (loss,) + outputs
            
        return outputs






class MAG_BERT(nn.Module):
    def __init__(self, args):
        
        super(MAG_BERT, self).__init__()
        
        self.model = MAG_BertForSequenceClassification.from_pretrained("/public/home/202420144954/bert-large-uncased", cache_dir = args.cache_path, args = args)
    
    def forward(self, text_feats, video_feats, audio_feats):

        outputs = self.model(
            text = text_feats,
            visual = video_feats,
            acoustic = audio_feats
        )
        
        return outputs
    