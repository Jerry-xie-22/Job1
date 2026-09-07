import os
import logging
import csv
from torch.utils.data import DataLoader

from .mm_pre import MMDataset
from .text_pre import TextDataset
from .video_pre import VideoDataset
from .audio_pre import AudioDataset
from .mm_pre import MMDataset
from .__init__ import benchmarks

__all__ = ['DataManager']

class DataManager:
    
    def __init__(self, args, logger_name = 'Multimodal Intent Recognition'):
        
        self.logger = logging.getLogger(logger_name)
        # 根据数据集名称获取基准配置intent_labels，binary_maps，binary_intent_labels，max_seq_lengths，feat_dims
        self.benchmarks = benchmarks[args.dataset]
        # 根据数据集名称获取数据集路径
        self.data_path = os.path.join(args.data_path, args.dataset)

        if args.data_mode == 'multi-class':
            self.label_list = self.benchmarks["intent_labels"]
        elif args.data_mode == 'binary-class': 
            self.label_list = self.benchmarks['binary_intent_labels']
        else:
            raise ValueError('The input data mode is not supported.')
        self.logger.info('Lists of intent labels are: %s', str(self.label_list))
        # 获取标签数量
        args.num_labels = len(self.label_list)
        args.label_list = self.label_list  
        #  设置特征维度和序列长度参数，文本维度768，语音256，视频768,下面文本序列长度是30，语音序列长度是230，视频序列长度是480
        args.text_feat_dim, args.video_feat_dim, args.audio_feat_dim = \
            self.benchmarks['feat_dims']['text'], self.benchmarks['feat_dims']['video'], self.benchmarks['feat_dims']['audio']
        args.text_seq_len, args.video_seq_len, args.audio_seq_len = \
            self.benchmarks['max_seq_lengths']['text'], self.benchmarks['max_seq_lengths']['video'], self.benchmarks['max_seq_lengths']['audio']
        # 获取index数组和标签数组
        self.train_data_index, self.train_label_ids, self.train_data_text = self._get_indexes_annotations(os.path.join(self.data_path, 'train.tsv'), args.data_mode)
        self.dev_data_index, self.dev_label_ids, self.dev_data_text = self._get_indexes_annotations(os.path.join(self.data_path, 'dev.tsv'), args.data_mode)
        self.test_data_index, self.test_label_ids,self.test_data_text = self._get_indexes_annotations(os.path.join(self.data_path, 'test.tsv'), args.data_mode)
        # 单模态特征获取，分别获取文本、视频和音频的单模态特征，使用TextDataset、VideoDataset和AudioDataset类处理各自的特征
        # 实际上就是三个数组，分别存储文本、视频和音频的特征
        self.unimodal_feats = self._get_unimodal_feats(args, self._get_attrs())

        # 为训练集、验证集和测试集分别创建MMDataset对象，每个mm_dataset对象包含一个样本的文本、视频和音频的特征还有id，这些都为tensor类型
        self.mm_data = self._get_multimodal_data(args)
        
        # 为训练集、验证集和测试集创建DataLoader对象，方便管理数据集
        self.mm_dataloader = self._get_dataloader(args, self.mm_data)


    def _get_indexes_annotations(self, read_file_path, data_mode):

        label_map = {}
        # 枚举标签列表，生成标签与数字的映射，比如["Inform", "Question", "Request", ...]会变成{"Inform": 0, "Question": 1, "Request": 2, ...}以实现数字化表示
        for i, label in enumerate(self.label_list):
            label_map[label] = i

        # 读取数据集的tsv文件，第一行是表头，所以从第二行开始读取
        with open(read_file_path, 'r') as f:

            data = csv.reader(f, delimiter="\t")
            indexes = [] # 存储索引，比如"S05_E16_329"
            label_ids = [] # 存储标签对应的数字，比如"Inform" -> 0，"Question" -> 1，"Request" -> 2
            texts = []

            for i, line in enumerate(data):
                # # 跳过表头行
                if i == 0:
                    continue

                # 根据每行的前面三列内容，拼接成一个字符串作为索引，比如line = ["S05", "E16", "329", "apparently...", "Inform"]，会生成：index = "S05_E16_329"
                index = '_'.join([line[0], line[1], line[2]])
                indexes.append(index)
                texts.append(line[3])
                
                # line[4]是标签，这里通过label_map获取标签对应的数字，然后存到label_id里面
                if data_mode == 'multi-class':
                    label_id = label_map[line[4]]
                else:
                    label_id = label_map[self.benchmarks['binary_maps'][line[4]]]
                
                label_ids.append(label_id)

        return indexes, label_ids, texts






    #   def _get_indexes_annotations(self, args, read_file_path, data_mode):

    #     label_map = {}
    #     # 枚举标签列表，生成标签与数字的映射，比如["Inform", "Question", "Request", ...]会变成{"Inform": 0, "Question": 1, "Request": 2, ...}以实现数字化表示
    #     for i, label in enumerate(self.label_list):
    #         label_map[label] = i

    #     # 读取数据集的tsv文件，第一行是表头，所以从第二行开始读取
    #     with open(read_file_path, 'r') as f:

    #         data = csv.reader(f, delimiter="\t")
    #         indexes = [] # 存储索引，比如"S05_E16_329"
    #         label_ids = [] # 存储标签对应的数字，比如"Inform" -> 0，"Question" -> 1，"Request" -> 2

    #         for i, line in enumerate(data):
    #             # # 跳过表头行
    #             if i == 0:
    #                 continue
                
    #             # 根据每行的前面三列内容，拼接成一个字符串作为索引，比如line = ["S05", "E16", "329", "apparently...", "Inform"]，会生成：index = "S05_E16_329"
    #             if args.dataset in ['MIntRec']:
    #                 index = '_'.join([line[0], line[1], line[2]])
    #                 indexes.append(index)
                
    #                 # line[4]是标签，这里通过label_map获取标签对应的数字，然后存到label_id里面
    #                 if data_mode == 'multi-class':
    #                     label_id = label_map[line[4]]
    #                 else:
    #                     label_id = label_map[self.benchmarks['binary_maps'][line[4]]]

    #             # elif args.dataset in ['MELD']:
    #             #     index = '_'.join([line[0], line[1]])
    #             #     indexes.append(index)
    #             #     if data_mode == 'multi-class':
    #             #         label_id = label_map[bm['label_maps'][line[3]]]
    #             #     else:
                
    
    #         label_ids.append(label_id)

    #     return indexes, label_ids
    
    def _get_unimodal_feats(self, args, attrs):
        
        text_feats = TextDataset(args, attrs).feats
        video_feats = VideoDataset(args, attrs).feats
        audio_feats = AudioDataset(args, attrs).feats

        return {
            'text': text_feats,
            'video': video_feats,
            'audio': audio_feats
        }
    
    def _get_multimodal_data(self, args):

        text_data = self.unimodal_feats['text']
        video_data = self.unimodal_feats['video']
        audio_data = self.unimodal_feats['audio']
        
        mm_train_data = MMDataset(self.train_label_ids, text_data['train'], video_data['train'], audio_data['train'], self.train_data_index, self.train_data_text)
        mm_dev_data = MMDataset(self.dev_label_ids, text_data['dev'], video_data['dev'], audio_data['dev'], self.dev_data_index, self.dev_data_text)
        mm_test_data = MMDataset(self.test_label_ids, text_data['test'], video_data['test'], audio_data['test'], self.test_data_index, self.test_data_text)

        return {
            'train': mm_train_data,
            'dev': mm_dev_data,
            'test': mm_test_data
        }

    def _get_dataloader(self, args, data):
        
        self.logger.info('Generate Dataloader Begin...')

        train_dataloader = DataLoader(data['train'], shuffle=True, batch_size = args.train_batch_size, num_workers = args.num_workers, pin_memory = True)
        dev_dataloader = DataLoader(data['dev'], batch_size = args.eval_batch_size, num_workers = args.num_workers, pin_memory = True)
        test_dataloader = DataLoader(data['test'], batch_size = args.eval_batch_size, num_workers = args.num_workers, pin_memory = True)
        
        self.logger.info('Generate Dataloader Finished...')

        return {
            'train': train_dataloader,
            'dev': dev_dataloader,
            'test': test_dataloader
        }
    
    # 获取类的所有属性
    def _get_attrs(self):

        attrs = {}
        for name, value in vars(self).items():
            attrs[name] = value

        return attrs


