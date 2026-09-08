import torch
import torch.nn.functional as F
import logging
from torch import nn, optim
from utils.functions import restore_model, save_model, EarlyStopping
from tqdm import trange, tqdm
from utils.metrics import AverageMeter, Metrics
from transformers import AdamW, get_linear_schedule_with_warmup
from .loss import TripletContrastiveLoss,SingleContrastiveLoss,KContrastiveLoss
from collections import defaultdict
from .LabelAttentionClassifier import MoELabelAttentionClassifier

__all__ = ['MAG_BERT']

class MAG_BERT:

    def __init__(self, args, data, model):

        self.logger = logging.getLogger(args.logger_name)
        
        self.device, self.model = model.device, model.model
        
        

        self.train_dataloader, self.eval_dataloader, self.test_dataloader = \
            data.mm_dataloader['train'], data.mm_dataloader['dev'], data.mm_dataloader['test']
        
        self.args = args
        self.criterion = nn.CrossEntropyLoss()
        self.label_cons_criterion = TripletContrastiveLoss(device = self.device, temperature=0.07)
        self.metrics = Metrics(args)
        self.label_classifier = MoELabelAttentionClassifier(
            num_experts=args.num_experts,
            hidden_dim=1024
        ).to(self.device)

        self.optimizer, self.scheduler = self._set_optimizer(args, data, self.model)

        if args.train:
            self.best_eval_score = 0
        else:
            self.model = restore_model(self.model, args.model_output_path)

    def _set_optimizer(self, args, data, model):
        
        param_optimizer = list(model.named_parameters())+ list(self.label_classifier.named_parameters()) + list(self.label_cons_criterion.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight','logit_scales', 'loss_scale']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        
        optimizer = AdamW(optimizer_grouped_parameters, lr = args.lr, correct_bias=False)
        num_train_examples = len(data.train_data_index)
        num_train_optimization_steps = int(num_train_examples / args.train_batch_size) * args.num_train_epochs
        num_warmup_steps= int(num_train_examples * args.num_train_epochs * args.warmup_proportion / args.train_batch_size)
        
        scheduler = get_linear_schedule_with_warmup(optimizer,
                                                    num_warmup_steps=num_warmup_steps,
                                                    num_training_steps=num_train_optimization_steps)
        
        return optimizer, scheduler

    def _train(self, args): 
        
        early_stopping = EarlyStopping(args)
        
        for epoch in trange(int(args.num_train_epochs), desc="Epoch"):
            self.model.train()
            loss_record = AverageMeter()
            cls_loss_record = AverageMeter()
            label_cons_loss_record = AverageMeter()
            
            for step, batch in enumerate(tqdm(self.train_dataloader, desc="Iteration")):
                # torch.Size([16, 3, 30])
                text_feats = batch['text_feats'].to(self.device)
                # torch.Size([16, 230, 256])
                video_feats = batch['video_feats'].to(self.device)
                # torch.Size([16, 480, 768])
                audio_feats = batch['audio_feats'].to(self.device)
                # torch.Size([16])
                label_ids = batch['label_ids'].to(self.device)

                with torch.set_grad_enabled(True):

                    outputs = self.model(text_feats, video_feats, audio_feats)
                    # 输出概率分布
                    logits = outputs[0]
                    # 输出句子向量
                    cls_output = outputs[1]

                    # cls_loss = self.criterion(logits, label_ids)

                    # 加载标签描述向量
                    label_description_embedding = torch.load("/public/home/202420144954/MIntRec-TCLMAP/MIntRec2.0/label_descriptions_mintrec2.0.pt") 
                    intent_to_embeddings = defaultdict(list)
                    for item in label_description_embedding.values():
                        intent = item["intent"]
                        emb = item["embedding"]  # shape: [L, D]
                        intent_to_embeddings[intent].append(torch.tensor(emb))
                    # 2. 保证每个 intent 恰好有 3 个描述
                    for k, v in intent_to_embeddings.items():
                        assert len(v) == 3, f"Intent {k} does not have exactly 3 embeddings"
                    
                    # 基于标签描述计算分类损失
                    all_label_embeds = []
                    for label_idx in range(args.num_labels):
                        # [3, D]
                        embs = torch.stack(intent_to_embeddings[label_idx], dim=0).to(self.device)
                        all_label_embeds.append(embs)
                    all_label_embeds = torch.stack(all_label_embeds, dim=0)  # [num_labels, 3, D]
                    # cls_output_norm = F.normalize(cls_output, dim=-1)  # 归一化 [B, D]
                    # label_embeds_norm = F.normalize(all_label_embeds, dim=-1)  # 归一化 [num_labels, 3, D]
                    # sim = torch.einsum('bd, lcd -> blc', cls_output_norm, label_embeds_norm) # 相似度计算: [B, num_labels, 3]
                    # sim_scores = sim.sum(dim=-1)  # 每个标签的三个描述向量相似度求和 [B, num_labels]
                    sim_scores, weights = self.label_classifier(cls_output, all_label_embeds)
                    cls_loss = self.criterion(sim_scores, label_ids) # 交叉熵分类损失

                    # 3. 构造 [T, 3, D] 张量
                    all_intents = sorted(intent_to_embeddings.keys())   # 保证固定顺序
                    label_embed_all = []
                    for intent in all_intents:
                        emb_list = intent_to_embeddings[intent]   # list of 3 tensors, 每个 [1, D]
                        emb_list = [e.view(1, -1) for e in emb_list]
                        stacked = torch.cat(emb_list, dim=0)      # [3, D]
                        label_embed_all.append(stacked)

                    label_embed = torch.stack(label_embed_all, dim=0).to(self.device)  # [T, 3, D]
                    # print("cls_output:", cls_output.shape)   # 期望 [B, D]
                    # print("label_desc:", label_embed.shape)   # 期望 [T, 3, D]

                    label_cons_loss = self.label_cons_criterion(cls_output, label_embed,label_ids)


                    loss = cls_loss + label_cons_loss
                    # loss = cls_loss

                    self.optimizer.zero_grad()

                    loss.backward()
                    loss_record.update(loss.item(), label_ids.size(0))
                    cls_loss_record.update(cls_loss.item(), label_ids.size(0))
                    label_cons_loss_record.update(label_cons_loss.item(),label_ids.size(0))
                    
                    self.optimizer.step()
                    self.scheduler.step()
            
            outputs = self._get_outputs(args, mode = 'eval')
            eval_score = outputs[args.eval_monitor]

            eval_results = {
                'train_loss': round(loss_record.avg, 4),
                'cls_loss': round(cls_loss_record.avg, 4),
                'label_cons_loss': round(label_cons_loss_record.avg, 4),
                'best_eval_score': round(early_stopping.best_score, 4),
                'eval_score': round(eval_score, 4)
            }

            self.logger.info("***** Epoch: %s: Eval results *****", str(epoch + 1))
            for key in eval_results.keys():
                self.logger.info("  %s = %s", key, str(eval_results[key]))
         
            early_stopping(eval_score, self.model)

            if early_stopping.early_stop:
                self.logger.info(f'EarlyStopping at epoch {epoch + 1}')
                break

        self.best_eval_score = early_stopping.best_score
        self.model = early_stopping.best_model   
        
        if args.save_model:
            self.logger.info('Trained models are saved in %s', args.model_output_path)
            save_model(self.model, args.model_output_path)   

    def _get_outputs(self, args, mode = 'eval', return_sample_results = False, show_results = False):
        
        if mode == 'eval':
            dataloader = self.eval_dataloader
        elif mode == 'test':
            dataloader = self.test_dataloader
        elif mode == 'train':
            dataloader = self.train_dataloader

        self.model.eval()

        total_labels = torch.empty(0,dtype=torch.long).to(self.device)
        total_preds = torch.empty(0,dtype=torch.long).to(self.device)
        total_logits = torch.empty((0, args.num_labels)).to(self.device)
        
        loss_record = AverageMeter()

        for batch in tqdm(dataloader, desc="Iteration"):

            text_feats = batch['text_feats'].to(self.device)
            video_feats = batch['video_feats'].to(self.device)
            audio_feats = batch['audio_feats'].to(self.device)
            label_ids = batch['label_ids'].to(self.device)
            
            with torch.set_grad_enabled(False):
                
                output = self.model(text_feats, video_feats, audio_feats)
                logits = output[0]
                cls_output = output[1]

                # 加载标签描述向量
                label_description_embedding = torch.load("/public/home/202420144954/MIntRec-TCLMAP/MIntRec2.0/label_descriptions_mintrec2.0.pt") 
                intent_to_embeddings = defaultdict(list)
                for item in label_description_embedding.values():
                    intent = item["intent"]
                    emb = item["embedding"]  # shape: [L, D]
                    intent_to_embeddings[intent].append(torch.tensor(emb))
                # 2. 保证每个 intent 恰好有 3 个描述
                for k, v in intent_to_embeddings.items():
                    assert len(v) == 3, f"Intent {k} does not have exactly 3 embeddings"
                    
                # 基于标签描述计算分类损失
                all_label_embeds = []
                for label_idx in range(args.num_labels):
                    # [3, D]
                    embs = torch.stack(intent_to_embeddings[label_idx], dim=0).to(self.device)
                    all_label_embeds.append(embs)

                all_label_embeds = torch.stack(all_label_embeds, dim=0)  # [num_labels, 3, D]
                # cls_output_norm = F.normalize(cls_output, dim=-1)  # 归一化 [B, D]
                # label_embeds_norm = F.normalize(all_label_embeds, dim=-1)  # 归一化 [num_labels, 3, D]
                # sim = torch.einsum('bd, lcd -> blc', cls_output_norm, label_embeds_norm) # 相似度计算: [B, num_labels, 3]
                # sim_scores = sim.sum(dim=-1)  # 每个标签的三个描述向量相似度求和 [B, num_labels]
                
                sim_scores, weights = self.label_classifier(cls_output, all_label_embeds)
                

                # total_logits = torch.cat((total_logits, logits))
                total_logits = torch.cat((total_logits, sim_scores))

                total_labels = torch.cat((total_labels, label_ids))
 

                # loss = self.criterion(logits, label_ids)
                loss = self.criterion(sim_scores, label_ids) 


                loss_record.update(loss.item(), label_ids.size(0))
                
        total_probs = F.softmax(total_logits.detach(), dim=1)
        total_maxprobs, total_preds = total_probs.max(dim = 1)

        y_pred = total_preds.cpu().numpy()
        y_true = total_labels.cpu().numpy()

        outputs = self.metrics(y_true, y_pred, show_results=show_results)
        outputs.update({'loss': loss_record.avg})

        if return_sample_results:

            outputs.update(
                {
                    'y_true': y_true,
                    'y_pred': y_pred
                }
            )

        return outputs

    def _test(self, args):

        test_results = self._get_outputs(args, mode = 'test', return_sample_results=True, show_results = True)
        test_results['best_eval_score'] = round(self.best_eval_score, 4)

        return test_results

























# class MAG_BERT:

#     def __init__(self, args, data, model):

#         self.logger = logging.getLogger(args.logger_name)
        
#         self.device, self.model = model.device, model.model
        
#         self.optimizer, self.scheduler = self._set_optimizer(args, data, self.model)

#         self.train_dataloader, self.eval_dataloader, self.test_dataloader = \
#             data.mm_dataloader['train'], data.mm_dataloader['dev'], data.mm_dataloader['test']
        
#         self.args = args
#         self.criterion = nn.CrossEntropyLoss()
#         self.label_cons_criterion = SingleContrastiveLoss(device = self.device, temperature=0.07)
#         self.metrics = Metrics(args)

#         if args.train:
#             self.best_eval_score = 0
#         else:
#             self.model = restore_model(self.model, args.model_output_path)

#     def _set_optimizer(self, args, data, model):
        
#         param_optimizer = list(model.named_parameters())
#         no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
#         optimizer_grouped_parameters = [
#             {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
#             {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
#         ]
        
#         optimizer = AdamW(optimizer_grouped_parameters, lr = args.lr, correct_bias=False)
#         num_train_examples = len(data.train_data_index)
#         num_train_optimization_steps = int(num_train_examples / args.train_batch_size) * args.num_train_epochs
#         num_warmup_steps= int(num_train_examples * args.num_train_epochs * args.warmup_proportion / args.train_batch_size)
        
#         scheduler = get_linear_schedule_with_warmup(optimizer,
#                                                     num_warmup_steps=num_warmup_steps,
#                                                     num_training_steps=num_train_optimization_steps)
        
#         return optimizer, scheduler

#     def _train(self, args): 
        
#         early_stopping = EarlyStopping(args)
        
#         for epoch in trange(int(args.num_train_epochs), desc="Epoch"):
#             self.model.train()
#             loss_record = AverageMeter()
#             cls_loss_record = AverageMeter()
#             label_cons_loss_record = AverageMeter()
            
#             for step, batch in enumerate(tqdm(self.train_dataloader, desc="Iteration")):
#                 # torch.Size([16, 3, 30])
#                 text_feats = batch['text_feats'].to(self.device)
#                 # torch.Size([16, 230, 256])
#                 video_feats = batch['video_feats'].to(self.device)
#                 # torch.Size([16, 480, 768])
#                 audio_feats = batch['audio_feats'].to(self.device)
#                 # torch.Size([16])
#                 label_ids = batch['label_ids'].to(self.device)

#                 with torch.set_grad_enabled(True):

#                     outputs = self.model(text_feats, video_feats, audio_feats)
#                     # 输出概率分布
#                     logits = outputs[0]
#                     # 输出句子向量
#                     cls_output = outputs[1]

#                     # cls_loss = self.criterion(logits, label_ids)

#                     # 加载标签描述向量
#                     label_description_embedding = torch.load("/home/cike/experiment_code/Base/data/label_descriptions_mintrec_with_body.pt") 
#                     intent_to_embeddings = defaultdict(list)
#                     for item in label_description_embedding.values():
#                         intent = item["intent"]
#                         emb = item["embedding"]  # shape: [L, D]
#                         intent_to_embeddings[intent].append(torch.tensor(emb))
#                     # 2. 保证每个 intent 恰好有 1 个描述
#                     for k, v in intent_to_embeddings.items():
#                         assert len(v) == 1, f"Intent {k} does not have exactly 1 embeddings"
                    
#                     # 基于标签描述计算分类损失（每个标签只有一个描述）
#                     all_label_embeds = []

#                     for label_idx in range(args.num_labels):
#                         # intent_to_embeddings[label_idx] 现在应该是一个向量而不是多个向量
#                         # 如果 intent_to_embeddings[label_idx] 是 shape [D]
#                         embs = intent_to_embeddings[label_idx][0].to(self.device) # [D]
#                         all_label_embeds.append(embs)

#                     # [num_labels, D]
#                     all_label_embeds = torch.stack(all_label_embeds, dim=0)

#                     # 归一化
#                     cls_output_norm = F.normalize(cls_output, dim=-1)          # [B, D]
#                     label_embeds_norm = F.normalize(all_label_embeds, dim=-1)  # [num_labels, D]
#                     print(cls_output_norm.shape)
#                     print(label_embeds_norm.shape)

#                     # 相似度：每个 B 和每个 label 做点积
#                     sim_scores = torch.einsum('bd, ld -> bl', cls_output_norm, label_embeds_norm)
#                     # sim_scores: [B, num_labels]
#                     # 分类损失
#                     cls_loss = self.criterion(sim_scores, label_ids)



#                     # 3. 构造 [T, 3, D] 张量,这里是构造所有标签的embedding，对比目标应该是
#                     all_intents = sorted(intent_to_embeddings.keys())   # 保证固定顺序
#                     label_embed_all = []
#                     for intent in all_intents:
#                         emb_list = intent_to_embeddings[intent]   # list of 3 tensors, 每个 [1, D]
#                         emb_list = [e.view(1, -1) for e in emb_list]
#                         stacked = torch.cat(emb_list, dim=0)      # [3, D]
#                         label_embed_all.append(stacked)

#                     label_embed = torch.stack(label_embed_all, dim=0).to(self.device)  # [T, 3, D]
#                     # print("cls_output:", cls_output.shape)   # 期望 [B, D]
#                     # print("label_desc:", label_embed.shape)   # 期望 [T, 3, D]


                    
#                     # # 计算标签对比损失， 这里是构造batch内所有标签的embedding
#                     # batch_size = label_ids.size(0)
#                     # label_embed = []
#                     # for i in range(batch_size):
#                     #     intent = label_ids[i].item()
#                     #     embeddings = intent_to_embeddings[intent]  # list of 3 tensors [768]
#                     #     stacked = torch.stack(embeddings, dim=0).unsqueeze(1).to(self.device)  # [3, 1, 768]
#                     #     label_embed.append(stacked)
#                     # label_embed = torch.stack(label_embed, dim=0)  # [B, 3, 1, 768]

#                     label_cons_loss = self.label_cons_criterion(cls_output, label_embed,label_ids)


#                     loss = cls_loss + label_cons_loss
#                     # loss = cls_loss

#                     self.optimizer.zero_grad()

#                     loss.backward()
#                     loss_record.update(loss.item(), label_ids.size(0))
#                     cls_loss_record.update(cls_loss.item(), label_ids.size(0))
#                     label_cons_loss_record.update(label_cons_loss.item(),label_ids.size(0))
                    
#                     self.optimizer.step()
#                     self.scheduler.step()
            
#             outputs = self._get_outputs(args, mode = 'eval')
#             eval_score = outputs[args.eval_monitor]

#             eval_results = {
#                 'train_loss': round(loss_record.avg, 4),
#                 'cls_loss': round(cls_loss_record.avg, 4),
#                 'label_cons_loss': round(label_cons_loss_record.avg, 4),
#                 'best_eval_score': round(early_stopping.best_score, 4),
#                 'eval_score': round(eval_score, 4)
#             }

#             self.logger.info("***** Epoch: %s: Eval results *****", str(epoch + 1))
#             for key in eval_results.keys():
#                 self.logger.info("  %s = %s", key, str(eval_results[key]))
         
#             early_stopping(eval_score, self.model)

#             if early_stopping.early_stop:
#                 self.logger.info(f'EarlyStopping at epoch {epoch + 1}')
#                 break

#         self.best_eval_score = early_stopping.best_score
#         self.model = early_stopping.best_model   
        
#         if args.save_model:
#             self.logger.info('Trained models are saved in %s', args.model_output_path)
#             save_model(self.model, args.model_output_path)   

#     def _get_outputs(self, args, mode = 'eval', return_sample_results = False, show_results = False):
        
#         if mode == 'eval':
#             dataloader = self.eval_dataloader
#         elif mode == 'test':
#             dataloader = self.test_dataloader
#         elif mode == 'train':
#             dataloader = self.train_dataloader

#         self.model.eval()

#         total_labels = torch.empty(0,dtype=torch.long).to(self.device)
#         total_preds = torch.empty(0,dtype=torch.long).to(self.device)
#         total_logits = torch.empty((0, args.num_labels)).to(self.device)
        
#         loss_record = AverageMeter()

#         for batch in tqdm(dataloader, desc="Iteration"):

#             text_feats = batch['text_feats'].to(self.device)
#             video_feats = batch['video_feats'].to(self.device)
#             audio_feats = batch['audio_feats'].to(self.device)
#             label_ids = batch['label_ids'].to(self.device)
            
#             with torch.set_grad_enabled(False):
                
#                 output = self.model(text_feats, video_feats, audio_feats)
#                 logits = output[0]
#                 cls_output = output[1]

#                 # 加载标签描述向量
#                 label_description_embedding = torch.load("/home/cike/experiment_code/Base/data/label_descriptions_mintrec_with_body.pt") 
#                 intent_to_embeddings = defaultdict(list)
#                 for item in label_description_embedding.values():
#                     intent = item["intent"]
#                     emb = item["embedding"]  # shape: [L, D]
#                     intent_to_embeddings[intent].append(torch.tensor(emb))
#                 # 2. 保证每个 intent 恰好有 3 个描述
#                 for k, v in intent_to_embeddings.items():
#                     assert len(v) == 1, f"Intent {k} does not have exactly 1 embeddings"
                    
#                 # 基于标签描述计算分类损失（每个标签只有一个描述）
#                 all_label_embeds = []

#                 for label_idx in range(args.num_labels):
#                     # intent_to_embeddings[label_idx] 现在应该是一个向量而不是多个向量
#                     # 如果 intent_to_embeddings[label_idx] 是 shape [D]
#                     embs = intent_to_embeddings[label_idx][0].to(self.device) # [D]
#                     all_label_embeds.append(embs)

#                 # [num_labels, D]
#                 all_label_embeds = torch.stack(all_label_embeds, dim=0)

#                 # 归一化
#                 cls_output_norm = F.normalize(cls_output, dim=-1)          # [B, D]
#                 label_embeds_norm = F.normalize(all_label_embeds, dim=-1)  # [num_labels, D]

#                 # 相似度：每个 B 和每个 label 做点积
#                 sim_scores = torch.einsum('bd, ld -> bl', cls_output_norm, label_embeds_norm)
#                 # sim_scores: [B, num_labels]
                


#                 # total_logits = torch.cat((total_logits, logits))
#                 total_logits = torch.cat((total_logits, sim_scores))
#                 total_labels = torch.cat((total_labels, label_ids))
 

#                 # loss = self.criterion(logits, label_ids)
#                 loss = self.criterion(sim_scores, label_ids) 


#                 loss_record.update(loss.item(), label_ids.size(0))
                
#         total_probs = F.softmax(total_logits.detach(), dim=1)
#         total_maxprobs, total_preds = total_probs.max(dim = 1)

#         y_pred = total_preds.cpu().numpy()
#         y_true = total_labels.cpu().numpy()

#         outputs = self.metrics(y_true, y_pred, show_results=show_results)
#         outputs.update({'loss': loss_record.avg})

#         if return_sample_results:

#             outputs.update(
#                 {
#                     'y_true': y_true,
#                     'y_pred': y_pred
#                 }
#             )

#         return outputs

#     def _test(self, args):

#         test_results = self._get_outputs(args, mode = 'test', return_sample_results=True, show_results = True)
#         test_results['best_eval_score'] = round(self.best_eval_score, 4)
    
#         return test_results
    
























    
# class MAG_BERT:

#     def __init__(self, args, data, model):

#         self.logger = logging.getLogger(args.logger_name)
        
#         self.device, self.model = model.device, model.model
        
#         self.optimizer, self.scheduler = self._set_optimizer(args, data, self.model)

#         self.train_dataloader, self.eval_dataloader, self.test_dataloader = \
#             data.mm_dataloader['train'], data.mm_dataloader['dev'], data.mm_dataloader['test']
        
#         self.args = args
#         self.criterion = nn.CrossEntropyLoss()
#         self.label_cons_criterion = KContrastiveLoss(device = self.device, temperature=0.07)
#         self.metrics = Metrics(args)

#         if args.train:
#             self.best_eval_score = 0
#         else:
#             self.model = restore_model(self.model, args.model_output_path)

#     def _set_optimizer(self, args, data, model):
        
#         param_optimizer = list(model.named_parameters())
#         no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
#         optimizer_grouped_parameters = [
#             {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
#             {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
#         ]
        
#         optimizer = AdamW(optimizer_grouped_parameters, lr = args.lr, correct_bias=False)
#         num_train_examples = len(data.train_data_index)
#         num_train_optimization_steps = int(num_train_examples / args.train_batch_size) * args.num_train_epochs
#         num_warmup_steps= int(num_train_examples * args.num_train_epochs * args.warmup_proportion / args.train_batch_size)
        
#         scheduler = get_linear_schedule_with_warmup(optimizer,
#                                                     num_warmup_steps=num_warmup_steps,
#                                                     num_training_steps=num_train_optimization_steps)
        
#         return optimizer, scheduler

#     def _train(self, args): 
        
#         early_stopping = EarlyStopping(args)
        
#         for epoch in trange(int(args.num_train_epochs), desc="Epoch"):
#             self.model.train()
#             loss_record = AverageMeter()
#             cls_loss_record = AverageMeter()
#             label_cons_loss_record = AverageMeter()
            
#             for step, batch in enumerate(tqdm(self.train_dataloader, desc="Iteration")):
#                 # torch.Size([16, 3, 30])
#                 text_feats = batch['text_feats'].to(self.device)
#                 # torch.Size([16, 230, 256])
#                 video_feats = batch['video_feats'].to(self.device)
#                 # torch.Size([16, 480, 768])
#                 audio_feats = batch['audio_feats'].to(self.device)
#                 # torch.Size([16])
#                 label_ids = batch['label_ids'].to(self.device)

#                 with torch.set_grad_enabled(True):

#                     outputs = self.model(text_feats, video_feats, audio_feats)
#                     # 输出概率分布
#                     logits = outputs[0]
#                     # 输出句子向量
#                     cls_output = outputs[1]

#                     # cls_loss = self.criterion(logits, label_ids)

#                     # 加载标签描述向量
#                     label_description_embedding = torch.load("/home/cike/experiment_code/Base/data/label_descriptions_mintrec_with_meaning_body.pt") 
#                     intent_to_embeddings = defaultdict(list)
#                     for item in label_description_embedding.values():
#                         intent = item["intent"]
#                         emb = item["embedding"]  # shape: [L, D]
#                         intent_to_embeddings[intent].append(torch.tensor(emb))
#                     # 2. 保证每个 intent 恰好有 2 个描述
#                     for k, v in intent_to_embeddings.items():
#                         assert len(v) == 2, f"Intent {k} does not have exactly 2 embeddings"
                    
#                     # 基于标签描述计算分类损失
#                     all_label_embeds = []
#                     for label_idx in range(args.num_labels):
#                         # [2, D]
#                         embs = torch.stack(intent_to_embeddings[label_idx], dim=0).to(self.device)
#                         all_label_embeds.append(embs)
#                     all_label_embeds = torch.stack(all_label_embeds, dim=0)  # [num_labels, 2, D]
#                     cls_output_norm = F.normalize(cls_output, dim=-1)  # 归一化 [B, D]
#                     label_embeds_norm = F.normalize(all_label_embeds, dim=-1)  # 归一化 [num_labels, 2, D]
#                     sim = torch.einsum('bd, lcd -> blc', cls_output_norm, label_embeds_norm) # 相似度计算: [B, num_labels, 2]
#                     sim_scores = sim.sum(dim=-1)  # 每个标签的三个描述向量相似度求和 [B, num_labels]
#                     cls_loss = self.criterion(sim_scores, label_ids) # 交叉熵分类损失

#                     # 3. 构造 [T, 2, D] 张量
#                     all_intents = sorted(intent_to_embeddings.keys())   # 保证固定顺序
#                     label_embed_all = []
#                     for intent in all_intents:
#                         emb_list = intent_to_embeddings[intent]   # list of 2 tensors, 每个 [1, D]
#                         emb_list = [e.view(1, -1) for e in emb_list]
#                         stacked = torch.cat(emb_list, dim=0)      # [2, D]
#                         label_embed_all.append(stacked)

#                     label_embed = torch.stack(label_embed_all, dim=0).to(self.device)  # [T, 2, D]
#                     # print("cls_output:", cls_output.shape)   # 期望 [B, D]
#                     # print("label_desc:", label_embed.shape)   # 期望 [T, 3, D]

#                     label_cons_loss = self.label_cons_criterion(cls_output, label_embed,label_ids)

#                     loss = cls_loss + label_cons_loss
#                     # loss = cls_loss

#                     self.optimizer.zero_grad()

#                     loss.backward()
#                     loss_record.update(loss.item(), label_ids.size(0))
#                     cls_loss_record.update(cls_loss.item(), label_ids.size(0))
#                     label_cons_loss_record.update(label_cons_loss.item(),label_ids.size(0))
                    
#                     self.optimizer.step()
#                     self.scheduler.step()
            
#             outputs = self._get_outputs(args, mode = 'eval')
#             eval_score = outputs[args.eval_monitor]

#             eval_results = {
#                 'train_loss': round(loss_record.avg, 4),
#                 'cls_loss': round(cls_loss_record.avg, 4),
#                 'label_cons_loss': round(label_cons_loss_record.avg, 4),
#                 'best_eval_score': round(early_stopping.best_score, 4),
#                 'eval_score': round(eval_score, 4)
#             }

#             self.logger.info("***** Epoch: %s: Eval results *****", str(epoch + 1))
#             for key in eval_results.keys():
#                 self.logger.info("  %s = %s", key, str(eval_results[key]))
         
#             early_stopping(eval_score, self.model)

#             if early_stopping.early_stop:
#                 self.logger.info(f'EarlyStopping at epoch {epoch + 1}')
#                 break

#         self.best_eval_score = early_stopping.best_score
#         self.model = early_stopping.best_model   
        
#         if args.save_model:
#             self.logger.info('Trained models are saved in %s', args.model_output_path)
#             save_model(self.model, args.model_output_path)   

#     def _get_outputs(self, args, mode = 'eval', return_sample_results = False, show_results = False):
        
#         if mode == 'eval':
#             dataloader = self.eval_dataloader
#         elif mode == 'test':
#             dataloader = self.test_dataloader
#         elif mode == 'train':
#             dataloader = self.train_dataloader

#         self.model.eval()

#         total_labels = torch.empty(0,dtype=torch.long).to(self.device)
#         total_preds = torch.empty(0,dtype=torch.long).to(self.device)
#         total_logits = torch.empty((0, args.num_labels)).to(self.device)
        
#         loss_record = AverageMeter()

#         for batch in tqdm(dataloader, desc="Iteration"):

#             text_feats = batch['text_feats'].to(self.device)
#             video_feats = batch['video_feats'].to(self.device)
#             audio_feats = batch['audio_feats'].to(self.device)
#             label_ids = batch['label_ids'].to(self.device)
            
#             with torch.set_grad_enabled(False):
                
#                 output = self.model(text_feats, video_feats, audio_feats)
#                 logits = output[0]
#                 cls_output = output[1]

#                 # 加载标签描述向量
#                 label_description_embedding = torch.load("/home/cike/experiment_code/Base/data/label_descriptions_mintrec_with_meaning_body.pt") 
#                 intent_to_embeddings = defaultdict(list)
#                 for item in label_description_embedding.values():
#                     intent = item["intent"]
#                     emb = item["embedding"]  # shape: [L, D]
#                     intent_to_embeddings[intent].append(torch.tensor(emb))
#                 # 2. 保证每个 intent 恰好有 2 个描述
#                 for k, v in intent_to_embeddings.items():
#                     assert len(v) == 2, f"Intent {k} does not have exactly 2 embeddings"
                    
#                 # 基于标签描述计算分类损失
#                 all_label_embeds = []
#                 for label_idx in range(args.num_labels):
#                     # [2, D]
#                     embs = torch.stack(intent_to_embeddings[label_idx], dim=0).to(self.device)
#                     all_label_embeds.append(embs)

#                 all_label_embeds = torch.stack(all_label_embeds, dim=0)  # [num_labels, 2, D]
#                 cls_output_norm = F.normalize(cls_output, dim=-1)  # 归一化 [B, D]
#                 label_embeds_norm = F.normalize(all_label_embeds, dim=-1)  # 归一化 [num_labels,2, D]
#                 sim = torch.einsum('bd, lcd -> blc', cls_output_norm, label_embeds_norm) # 相似度计算: [B, num_labels, 2]
#                 sim_scores = sim.sum(dim=-1)  # 每个标签的三个描述向量相似度求和 [B, num_labels]
                

#                 # total_logits = torch.cat((total_logits, logits))
#                 total_logits = torch.cat((total_logits, sim_scores))

#                 total_labels = torch.cat((total_labels, label_ids))
 

#                 # loss = self.criterion(logits, label_ids)
#                 loss = self.criterion(sim_scores, label_ids) 


#                 loss_record.update(loss.item(), label_ids.size(0))
                
#         total_probs = F.softmax(total_logits.detach(), dim=1)
#         total_maxprobs, total_preds = total_probs.max(dim = 1)

#         y_pred = total_preds.cpu().numpy()
#         y_true = total_labels.cpu().numpy()

#         outputs = self.metrics(y_true, y_pred, show_results=show_results)
#         outputs.update({'loss': loss_record.avg})

#         if return_sample_results:

#             outputs.update(
#                 {
#                     'y_true': y_true,
#                     'y_pred': y_pred
#                 }
#             )

#         return outputs

#     def _test(self, args):

#         test_results = self._get_outputs(args, mode = 'test', return_sample_results=True, show_results = True)
#         test_results['best_eval_score'] = round(self.best_eval_score, 4)
    
#         return test_results































# class MAG_BERT:

#     def __init__(self, args, data, model):

#         self.logger = logging.getLogger(args.logger_name)
        
#         self.device, self.model = model.device, model.model
        
#         self.optimizer, self.scheduler = self._set_optimizer(args, data, self.model)

#         self.train_dataloader, self.eval_dataloader, self.test_dataloader = \
#             data.mm_dataloader['train'], data.mm_dataloader['dev'], data.mm_dataloader['test']
        
#         self.args = args
#         self.criterion = nn.CrossEntropyLoss()
#         self.label_cons_criterion = TripletContrastiveLoss(device = self.device, temperature=0.07)
#         self.metrics = Metrics(args)
#         self.label_classifier = DegeneratedMoELabelClassifier(
#             num_experts=args.num_experts,
#             hidden_dim=768
#         ).to(self.device)

#         if args.train:
#             self.best_eval_score = 0
#         else:
#             self.model = restore_model(self.model, args.model_output_path)

#     def _set_optimizer(self, args, data, model):
        
#         param_optimizer = list(model.named_parameters())
#         no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
#         optimizer_grouped_parameters = [
#             {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
#             {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
#         ]
        
#         optimizer = AdamW(optimizer_grouped_parameters, lr = args.lr, correct_bias=False)
#         num_train_examples = len(data.train_data_index)
#         num_train_optimization_steps = int(num_train_examples / args.train_batch_size) * args.num_train_epochs
#         num_warmup_steps= int(num_train_examples * args.num_train_epochs * args.warmup_proportion / args.train_batch_size)
        
#         scheduler = get_linear_schedule_with_warmup(optimizer,
#                                                     num_warmup_steps=num_warmup_steps,
#                                                     num_training_steps=num_train_optimization_steps)
        
#         return optimizer, scheduler

#     def _train(self, args): 
        
#         early_stopping = EarlyStopping(args)
        
#         for epoch in trange(int(args.num_train_epochs), desc="Epoch"):
#             self.model.train()
#             loss_record = AverageMeter()
#             cls_loss_record = AverageMeter()
#             label_cons_loss_record = AverageMeter()
            
#             for step, batch in enumerate(tqdm(self.train_dataloader, desc="Iteration")):
#                 # torch.Size([16, 3, 30])
#                 text_feats = batch['text_feats'].to(self.device)
#                 # torch.Size([16, 230, 256])
#                 video_feats = batch['video_feats'].to(self.device)
#                 # torch.Size([16, 480, 768])
#                 audio_feats = batch['audio_feats'].to(self.device)
#                 # torch.Size([16])
#                 label_ids = batch['label_ids'].to(self.device)

#                 with torch.set_grad_enabled(True):

#                     outputs = self.model(text_feats, video_feats, audio_feats)
#                     # 输出概率分布
#                     logits = outputs[0]
#                     # 输出句子向量
#                     cls_output = outputs[1]

#                     # cls_loss = self.criterion(logits, label_ids)

#                     # 加载标签描述向量
#                     label_description_embedding = torch.load("/home/cike/experiment_code/Base/data/label_embeddings_with_info.pt") 
#                     intent_to_embeddings = defaultdict(list)
#                     for item in label_description_embedding.values():
#                         intent = item["intent"]
#                         emb = item["embedding"]  # shape: [L, D]
#                         intent_to_embeddings[intent].append(torch.tensor(emb))
#                     # 2. 保证每个 intent 恰好有 3 个描述
#                     for k, v in intent_to_embeddings.items():
#                         assert len(v) == 3, f"Intent {k} does not have exactly 3 embeddings"
                    
#                     # 基于标签描述计算分类损失
#                     all_label_embeds = []
#                     for label_idx in range(args.num_labels):
#                         # [3, D]
#                         embs = torch.stack(intent_to_embeddings[label_idx], dim=0).to(self.device)
#                         all_label_embeds.append(embs)
#                     all_label_embeds = torch.stack(all_label_embeds, dim=0)  # [num_labels, 3, D]
#                     cls_output_norm = F.normalize(cls_output, dim=-1)  # 归一化 [B, D]
#                     label_embeds_norm = F.normalize(all_label_embeds, dim=-1)  # 归一化 [num_labels, 3, D]
#                     sim = torch.einsum('bd, lcd -> blc', cls_output_norm, label_embeds_norm) # 相似度计算: [B, num_labels, 3]
#                     sim_scores = sim.sum(dim=-1)  # 每个标签的三个描述向量相似度求和 [B, num_labels]
#                     # sim_scores, att = self.label_classifier(cls_output, all_label_embeds)
#                     cls_loss = self.criterion(sim_scores, label_ids) # 交叉熵分类损失

#                     # 3. 构造 [T, 3, D] 张量
#                     all_intents = sorted(intent_to_embeddings.keys())   # 保证固定顺序
#                     label_embed_all = []
#                     for intent in all_intents:
#                         emb_list = intent_to_embeddings[intent]   # list of 3 tensors, 每个 [1, D]
#                         emb_list = [e.view(1, -1) for e in emb_list]
#                         stacked = torch.cat(emb_list, dim=0)      # [3, D]
#                         label_embed_all.append(stacked)

#                     label_embed = torch.stack(label_embed_all, dim=0).to(self.device)  # [T, 3, D]
#                     # print("cls_output:", cls_output.shape)   # 期望 [B, D]
#                     # print("label_desc:", label_embed.shape)   # 期望 [T, 3, D]


                    
#                     # 计算标签对比损失
#                     # batch_size = label_ids.size(0)
#                     # label_embed = []
#                     # for i in range(batch_size):
#                     #     intent = label_ids[i].item()
#                     #     embeddings = intent_to_embeddings[intent]  # list of 3 tensors [768]
#                     #     stacked = torch.stack(embeddings, dim=0).unsqueeze(1).to(self.device)  # [3, 1, 768]
#                     #     label_embed.append(stacked)
#                     # label_embed = torch.stack(label_embed, dim=0)  # [B, 3, 1, 768]
#                     # print(label_embed.shape)
#                     # print(cls_output.shape)
#                     # print(label_ids.shape)
#                     label_cons_loss = self.label_cons_criterion(cls_output, label_embed,label_ids)


#                     loss = cls_loss + label_cons_loss
#                     # loss = cls_loss

#                     self.optimizer.zero_grad()

#                     loss.backward()
#                     loss_record.update(loss.item(), label_ids.size(0))
#                     cls_loss_record.update(cls_loss.item(), label_ids.size(0))
#                     label_cons_loss_record.update(label_cons_loss.item(),label_ids.size(0))
                    
#                     self.optimizer.step()
#                     self.scheduler.step()
            
#             outputs = self._get_outputs(args, mode = 'eval')
#             eval_score = outputs[args.eval_monitor]

#             eval_results = {
#                 'train_loss': round(loss_record.avg, 4),
#                 'cls_loss': round(cls_loss_record.avg, 4),
#                 'label_cons_loss': round(label_cons_loss_record.avg, 4),
#                 'best_eval_score': round(early_stopping.best_score, 4),
#                 'eval_score': round(eval_score, 4)
#             }

#             self.logger.info("***** Epoch: %s: Eval results *****", str(epoch + 1))
#             for key in eval_results.keys():
#                 self.logger.info("  %s = %s", key, str(eval_results[key]))
         
#             early_stopping(eval_score, self.model)

#             if early_stopping.early_stop:
#                 self.logger.info(f'EarlyStopping at epoch {epoch + 1}')
#                 break

#         self.best_eval_score = early_stopping.best_score
#         self.model = early_stopping.best_model   
        
#         if args.save_model:
#             self.logger.info('Trained models are saved in %s', args.model_output_path)
#             save_model(self.model, args.model_output_path)   

#     def _get_outputs(self, args, mode = 'eval', return_sample_results = False, show_results = False):
        
#         if mode == 'eval':
#             dataloader = self.eval_dataloader
#         elif mode == 'test':
#             dataloader = self.test_dataloader
#         elif mode == 'train':
#             dataloader = self.train_dataloader

#         self.model.eval()

#         total_labels = torch.empty(0,dtype=torch.long).to(self.device)
#         total_preds = torch.empty(0,dtype=torch.long).to(self.device)
#         total_logits = torch.empty((0, args.num_labels)).to(self.device)
        
#         loss_record = AverageMeter()

#         for batch in tqdm(dataloader, desc="Iteration"):

#             text_feats = batch['text_feats'].to(self.device)
#             video_feats = batch['video_feats'].to(self.device)
#             audio_feats = batch['audio_feats'].to(self.device)
#             label_ids = batch['label_ids'].to(self.device)
            
#             with torch.set_grad_enabled(False):
                
#                 output = self.model(text_feats, video_feats, audio_feats)
#                 logits = output[0]
#                 cls_output = output[1]

#                 # 加载标签描述向量
#                 label_description_embedding = torch.load("/home/cike/experiment_code/Base/data/label_embeddings_with_info.pt") 
#                 intent_to_embeddings = defaultdict(list)
#                 for item in label_description_embedding.values():
#                     intent = item["intent"]
#                     emb = item["embedding"]  # shape: [L, D]
#                     intent_to_embeddings[intent].append(torch.tensor(emb))
#                 # 2. 保证每个 intent 恰好有 3 个描述
#                 for k, v in intent_to_embeddings.items():
#                     assert len(v) == 3, f"Intent {k} does not have exactly 3 embeddings"
                    
#                 # 基于标签描述计算分类损失
#                 all_label_embeds = []
#                 for label_idx in range(args.num_labels):
#                     # [3, D]
#                     embs = torch.stack(intent_to_embeddings[label_idx], dim=0).to(self.device)
#                     all_label_embeds.append(embs)

#                 all_label_embeds = torch.stack(all_label_embeds, dim=0)  # [num_labels, 3, D]
#                 cls_output_norm = F.normalize(cls_output, dim=-1)  # 归一化 [B, D]
#                 label_embeds_norm = F.normalize(all_label_embeds, dim=-1)  # 归一化 [num_labels, 3, D]
#                 sim = torch.einsum('bd, lcd -> blc', cls_output_norm, label_embeds_norm) # 相似度计算: [B, num_labels, 3]
#                 sim_scores = sim.sum(dim=-1)  # 每个标签的三个描述向量相似度求和 [B, num_labels]
#                 # sim_scores, att = self.label_classifier(cls_output, all_label_embeds)
                

#                 # total_logits = torch.cat((total_logits, logits))
#                 total_logits = torch.cat((total_logits, sim_scores))

#                 total_labels = torch.cat((total_labels, label_ids))
 

#                 # loss = self.criterion(logits, label_ids)
#                 loss = self.criterion(sim_scores, label_ids) 


#                 loss_record.update(loss.item(), label_ids.size(0))
                
#         total_probs = F.softmax(total_logits.detach(), dim=1)
#         total_maxprobs, total_preds = total_probs.max(dim = 1)

#         y_pred = total_preds.cpu().numpy()
#         y_true = total_labels.cpu().numpy()

#         outputs = self.metrics(y_true, y_pred, show_results=show_results)
#         outputs.update({'loss': loss_record.avg})

#         if return_sample_results:

#             outputs.update(
#                 {
#                     'y_true': y_true,
#                     'y_pred': y_pred
#                 }
#             )

#         return outputs

#     def _test(self, args):

#         test_results = self._get_outputs(args, mode = 'test', return_sample_results=True, show_results = True)
#         test_results['best_eval_score'] = round(self.best_eval_score, 4)
    
#         return test_results