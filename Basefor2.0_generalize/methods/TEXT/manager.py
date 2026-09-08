import torch
import torch.nn.functional as F
import logging
from torch import nn
from tqdm import trange, tqdm
from transformers import BertForSequenceClassification
from utils.functions import restore_model, save_model, EarlyStopping
from utils.metrics import AverageMeter, Metrics
from torch.utils.data import Dataset
from torch import optim
from transformers import AdamW, get_linear_schedule_with_warmup

__all__ = ['TEXT']

class TextDataset(Dataset):
    
    def __init__(self, label_ids, text_feats):
        
        self.label_ids = torch.tensor(label_ids)
        self.text_feats = torch.tensor(text_feats)
        self.size = len(self.text_feats)

    def __len__(self):
        return self.size

    def __getitem__(self, index):

        sample = {
            'text_feats': self.text_feats[index],
            'label_ids': self.label_ids[index], 
        } 
        return sample

class TEXT:

    def __init__(self, args, data):
        # 又是设置日志记录，logger的输出都会输出到这个文件
        self.logger = logging.getLogger(args.logger_name)
        # 加载bert预训练模型，BertForSequenceClassification是一个用于文本分类任务的 BERT 模型封装类
        self.model = BertForSequenceClassification.from_pretrained(
            args.text_backbone, # 模型名称
            cache_dir = args.cache_path,  # 预训练模型的缓存目录，如果该路径有模型就不用下载，没有就要下载
            num_labels = args.num_labels) # 分类任务的标签数量
        # optimizer：优化器，就是用于梯度下载的优化器，比如adam，SGD
        # scheduler：学习率调度器，用于动态调整学习率
        self.optimizer, self.scheduler = self._set_optimizer(args, data, self.model)
        # 将模型移动到指定的计算设备上
        self.device = torch.device('cuda:%d' % int(args.gpu_id) if torch.cuda.is_available() else 'cpu')
        # 将模型移动到指定的计算设备上
        self.model.to(self.device)
        
        # # 创建数据集
        self.train_data = TextDataset(data.train_label_ids, data.unimodal_feats['text']['train'])
        self.dev_data = TextDataset(data.dev_label_ids, data.unimodal_feats['text']['dev'])
        self.test_data = TextDataset(data.test_label_ids, data.unimodal_feats['text']['test'])
        self.text_data = {
            'train': self.train_data,
            'dev': self.dev_data,
            'test': self.test_data
        }

        self.text_dataloader = data._get_dataloader(args, self.text_data)

        self.train_dataloader, self.eval_dataloader, self.test_dataloader = \
            self.text_dataloader['train'], self.text_dataloader['dev'], self.text_dataloader['test']
        
        self.criterion = nn.CrossEntropyLoss()
        self.metrics = Metrics(args)
        
        if args.train:
            self.best_eval_score = None
        else:
            self.model = restore_model(self.model, args.model_output_path)

    def _set_optimizer(self, args, data, model):

        # 拿到所有模型参数
        param_optimizer = list(model.named_parameters())
        # 把不做权重衰减（L2正则化）的参数拿出来，一般来说，bias 和 LayerNorm 层的参数不建议做权重衰减
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        # 参数分组，划分为两组，一组是需要权重衰减的参数，另一组是不需要权重衰减的参数
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        # 使用 AdamW 优化器，学习率设置为 args.lr
        optimizer = AdamW(optimizer_grouped_parameters, lr = args.lr, correct_bias=False)
        # 计算训练样本总数
        num_train_examples = len(data.train_data_index)
        # 计算训练的总步数
        num_train_optimization_steps = int(num_train_examples / args.train_batch_size) * args.num_train_epochs
        # 预热步数
        num_warmup_steps= int(num_train_examples * args.num_train_epochs * args.warmup_proportion / args.train_batch_size)
        # 设置学习率调度器，前 num_warmup_steps 线性升高学习率，然后线性下降到 0。
        scheduler = get_linear_schedule_with_warmup(optimizer,
                                                    num_warmup_steps=num_warmup_steps,
                                                    num_training_steps=num_train_optimization_steps)
        return optimizer, scheduler

    def _train(self, args): 
        # 用于在验证集性能不再提升时提前终止训练
        early_stopping = EarlyStopping(args)

        # 外层循环控制训练轮数。
        for epoch in trange(int(args.num_train_epochs), desc="Epoch"):
            self.model.train()
            # 自定义类，用于计算 epoch 内的平均损失
            loss_record = AverageMeter()
            
            # 对训练数据逐批训练
            for step, batch in enumerate(tqdm(self.train_dataloader, desc="Iteration")):

                text_feats = batch['text_feats'].to(self.device)
                label_ids = batch['label_ids'].to(self.device)

                with torch.set_grad_enabled(True):
                    input_ids, input_mask, segment_ids = text_feats[:, 0], text_feats[:, 1], text_feats[:, 2]
                    # 前向传播：输入是一个batch的输入
                    outputs = self.model(input_ids = input_ids, token_type_ids = segment_ids, attention_mask = input_mask)
                    # 计算一个batch的损失,self.criterion是定义好的损失函数
                    loss = self.criterion(outputs.logits, label_ids)
                    # 清空旧的梯度
                    self.optimizer.zero_grad()

                    # 反向传播
                    loss.backward()
                    # 记录损失
                    loss_record.update(loss.item(), label_ids.size(0))
                    
                    self.optimizer.step()
                    self.scheduler.step()            
                    
            outputs = self._get_outputs(args, mode = 'eval')
            eval_score = outputs[args.eval_monitor]

            eval_results = {
                'train_loss': round(loss_record.avg, 4),
                'best_eval_score': round(early_stopping.best_score, 4),
                'eval_score': round(eval_score, 4)
            }

            self.logger.info("***** Epoch: %s: Eval results *****", str(epoch + 1))
            for key in eval_results.keys():
                self.logger.info("  %s = %s", key, str(eval_results[key]))

            # EarlyStopping 判断是否终止
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
            label_ids = batch['label_ids'].to(self.device)
            
            with torch.set_grad_enabled(False):
                
                input_ids, input_mask, segment_ids = text_feats[:, 0], text_feats[:, 1], text_feats[:, 2]
                logits = self.model(input_ids = input_ids, token_type_ids = segment_ids, attention_mask = input_mask).logits
                
                total_logits = torch.cat((total_logits, logits))
                total_labels = torch.cat((total_labels, label_ids))

                loss = self.criterion(logits, label_ids)
                loss_record.update(loss.item(), label_ids.size(0))
      
        total_probs = F.softmax(total_logits.detach(), dim=1)
        total_maxprobs, total_preds = total_probs.max(dim = 1)

        y_pred = total_preds.cpu().numpy()
        y_true = total_labels.cpu().numpy()
        
        show_results = True if mode == 'test' else False

        outputs = self.metrics(y_true, y_pred, show_results = show_results)
        outputs.update({'eval_loss': loss_record.avg})

        if return_sample_results:
            outputs.update(
                {
                    'y_true': y_true,
                    'y_pred': y_pred
                }
            )

        return outputs




    def _test(self, args):

        test_results = self._get_outputs(args, mode = 'test', return_sample_results = True, show_results = True)
        test_results['best_eval_score'] = round(self.best_eval_score, 4)

        return test_results