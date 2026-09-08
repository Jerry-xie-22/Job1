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
from sklearn.metrics import precision_score, recall_score, f1_score
import csv
import os

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
        # job1
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
        
        # job1
        param_optimizer = list(model.named_parameters())+ list(self.label_classifier.named_parameters()) + list(self.label_cons_criterion.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight','logit_scales', 'loss_scale']
        param_optimizer = list(model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        
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

                    cls_loss = self.criterion(logits, label_ids)

                    # 加载标签描述向量
                    label_description_embedding = torch.load("/public/home/202420144954/job2/Base_for_emo_mintrec10_c2f_inject_large/data/label_descriptions_mintrec.pt") 
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

                    # job1废弃的不要的
                    # cls_output_norm = F.normalize(cls_output, dim=-1)  # 归一化 [B, D]
                    # label_embeds_norm = F.normalize(all_label_embeds, dim=-1)  # 归一化 [num_labels, 3, D]
                    # sim = torch.einsum('bd, lcd -> blc', cls_output_norm, label_embeds_norm) # 相似度计算: [B, num_labels, 3]
                    # sim_scores = sim.sum(dim=-1)  # 每个标签的三个描述向量相似度求和 [B, num_labels]

                    # job1
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

                    # job1
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

        all_texts = []
        all_indexes = []

        for batch in tqdm(dataloader, desc="Iteration"):

            text_feats = batch['text_feats'].to(self.device)
            video_feats = batch['video_feats'].to(self.device)
            audio_feats = batch['audio_feats'].to(self.device)
            label_ids = batch['label_ids'].to(self.device)

            batch_texts = batch.get('text', [''] * label_ids.size(0))
            batch_indexes = batch.get('sample_index', [''] * label_ids.size(0))
            
            with torch.set_grad_enabled(False):
                
                output = self.model(text_feats, video_feats, audio_feats)
                logits = output[0]
                cls_output = output[1]

                # 加载标签描述向量
                label_description_embedding = torch.load("/public/home/202420144954/job2/Base_for_emo_mintrec10_c2f_inject_large/data/label_descriptions_mintrec.pt") 
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

                # job1废弃的不要的
                # cls_output_norm = F.normalize(cls_output, dim=-1)  # 归一化 [B, D]
                # label_embeds_norm = F.normalize(all_label_embeds, dim=-1)  # 归一化 [num_labels, 3, D]
                # sim = torch.einsum('bd, lcd -> blc', cls_output_norm, label_embeds_norm) # 相似度计算: [B, num_labels, 3]
                # sim_scores = sim.sum(dim=-1)  # 每个标签的三个描述向量相似度求和 [B, num_labels]

                # job1
                sim_scores, weights = self.label_classifier(cls_output, all_label_embeds)
                
                # job1
                # total_logits = torch.cat((total_logits, logits))
                total_logits = torch.cat((total_logits, sim_scores))

                total_labels = torch.cat((total_labels, label_ids))
 
                # job1
                # loss = self.criterion(logits, label_ids)
                loss = self.criterion(sim_scores, label_ids) 


                loss_record.update(loss.item(), label_ids.size(0))

            if return_sample_results:
                if isinstance(batch_texts, torch.Tensor):
                    batch_texts = batch_texts.cpu().numpy().tolist()
                elif isinstance(batch_texts, (list, tuple)):
                    batch_texts = list(batch_texts)
                if isinstance(batch_indexes, torch.Tensor):
                    batch_indexes = batch_indexes.cpu().numpy().tolist()
                elif isinstance(batch_indexes, (list, tuple)):
                    batch_indexes = list(batch_indexes)
                all_texts.extend(batch_texts)
                all_indexes.extend(batch_indexes)     
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
                    'y_pred': y_pred,
                    'texts': all_texts,
                    'indexes': all_indexes
                }
            )

        return outputs

    def _test(self, args):

        test_results = self._get_outputs(args, mode = 'test', return_sample_results=True, show_results = True)
        test_results['best_eval_score'] = round(self.best_eval_score, 4)

        # 统计预测标签与真实标签
        y_true = test_results.get('y_true', [])
        y_pred = test_results.get('y_pred', [])
        texts = test_results.get('texts', [])
        indexes = test_results.get('indexes', [])

        label_list = args.label_list if hasattr(args, 'label_list') else None

        tsv_output_path = os.path.join(args.results_path, 'test_predictions_without_module.tsv')
        with open(tsv_output_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f, delimiter='\t')
            writer.writerow(['index', 'text', 'true_label', 'true_label_id', 'pred_label', 'pred_label_id'])
            for i in range(len(y_true)):
                true_label = label_list[y_true[i]] if label_list else str(y_true[i])
                pred_label = label_list[y_pred[i]] if label_list else str(y_pred[i])
                text = texts[i] if i < len(texts) else ''
                idx = indexes[i] if i < len(indexes) else ''
                writer.writerow([idx, text, true_label, y_true[i], pred_label, y_pred[i]])

        self.logger.info('Test predictions saved to %s', tsv_output_path)


        # ---- per-class metrics TSV ----
        labels = sorted(set(y_true))
        per_class_prec = precision_score(y_true, y_pred, average=None, labels=labels)
        per_class_rec = recall_score(y_true, y_pred, average=None, labels=labels)
        per_class_f1 = f1_score(y_true, y_pred, average=None, labels=labels)

        each_class_path = os.path.join(args.results_path, 'output_show_each_class.tsv')
        with open(each_class_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f, delimiter='\t')
            writer.writerow(['label_id', 'label_name', 'prec', 'rec', 'f1'])
            for idx, label_id in enumerate(labels):
                name = label_list[label_id] if label_list else str(label_id)
                writer.writerow([
                    label_id, name,
                    round(per_class_prec[idx], 4),
                    round(per_class_rec[idx], 4),
                    round(per_class_f1[idx], 4),
                ])
            writer.writerow([])
            writer.writerow(['overall', '', 'acc', 'f1', 'prec', 'rec', 'wprec', 'wrec', 'wf1'])
            writer.writerow([
                '', '',
                round(test_results['acc'], 4),
                round(test_results['f1'], 4),
                round(test_results['prec'], 4),
                round(test_results['rec'], 4),
                round(test_results['weighted_prec'], 4),
                round(test_results['weighted_rec'], 4),
                round(test_results['weighted_f1'], 4),
            ])

        self.logger.info('Per-class metrics saved to %s', each_class_path)

        return test_results




















