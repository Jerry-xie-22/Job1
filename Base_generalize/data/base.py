import copy
import json
import logging
import os

from torch.utils.data import DataLoader

from . import benchmarks
from .audio_pre import AudioDataset
from .generalize import read_annotations
from .mm_pre import MMDataset
from .text_pre import TextDataset
from .video_pre import VideoDataset

__all__ = ['DataManager']


class DataManager:
    def __init__(self, args, logger_name=None):
        self.logger = logging.getLogger(logger_name or args.logger_name)
        if args.dataset != 'MIntRec':
            raise ValueError('Base_generalize currently supports MIntRec -> MIntRec2.0 only')
        self.benchmarks = copy.deepcopy(benchmarks['MIntRec'])
        self.data_path = os.path.join(args.data_path, args.dataset)
        if args.data_mode == 'multi-class':
            self.label_list = self.benchmarks['intent_labels']
            binary_maps = None
        elif args.data_mode == 'binary-class':
            self.label_list = self.benchmarks['binary_intent_labels']
            binary_maps = self.benchmarks['binary_maps']
        else:
            raise ValueError('Unsupported data mode')
        args.num_labels = len(self.label_list)
        args.label_list = self.label_list
        self.benchmarks['max_seq_lengths'] = {'text': 50, 'video': 230, 'audio': 480}
        for modality in ('text', 'video', 'audio'):
            setattr(args, f'{modality}_seq_len', self.benchmarks['max_seq_lengths'][modality])
            setattr(args, f'{modality}_feat_dim', self.benchmarks['feat_dims'][modality])
        self.generalization_report = {
            'source_dataset': 'MIntRec', 'target_dataset': 'MIntRec2.0',
            'protocol': 'source-label closed-set evaluation', 'label_order': self.label_list,
            'splits': {},
        }
        for split in ('train', 'dev', 'test'):
            dataset = 'MIntRec2.0' if split == 'test' else 'MIntRec'
            path = os.path.join(self.data_path, f'{split}.tsv')
            indexes, labels, texts, skipped = read_annotations(
                path, dataset, self.label_list, binary_maps, filter_unknown=split == 'test')
            setattr(self, f'{split}_data_index', indexes)
            setattr(self, f'{split}_label_ids', labels)
            setattr(self, f'{split}_data_text', texts)
            total = len(indexes) + sum(skipped.values())
            self.generalization_report['splits'][split] = {
                'path': path, 'dataset': dataset, 'total': total, 'retained': len(indexes),
                'excluded_by_label': skipped, 'coverage': len(indexes) / total,
            }
            self.logger.info('%s (%s): retained %d/%d; excluded labels=%s',
                             split, dataset, len(indexes), total, skipped)
        attrs = vars(self)
        self.unimodal_feats = {
            'text': TextDataset(args, attrs).feats,
            'video': VideoDataset(args, attrs).feats,
            'audio': AudioDataset(args, attrs).feats,
        }
        self.mm_data = {
            split: MMDataset(getattr(self, f'{split}_label_ids'),
                             self.unimodal_feats['text'][split],
                             self.unimodal_feats['video'][split],
                             self.unimodal_feats['audio'][split],
                             getattr(self, f'{split}_data_index'),
                             getattr(self, f'{split}_data_text'))
            for split in ('train', 'dev', 'test')
        }
        batch_sizes = {'train': args.train_batch_size, 'dev': args.eval_batch_size,
                       'test': args.test_batch_size}
        self.mm_dataloader = {
            split: DataLoader(dataset, shuffle=split == 'train', batch_size=batch_sizes[split],
                              num_workers=args.num_workers, pin_memory=True)
            for split, dataset in self.mm_data.items()
        }
        os.makedirs(args.results_path, exist_ok=True)
        report_path = os.path.join(args.results_path, f'{args.logger_name}_data_report.json')
        with open(report_path, 'w', encoding='utf-8') as stream:
            json.dump(self.generalization_report, stream, ensure_ascii=False, indent=2)
        self.logger.info('Generalization data report: %s', report_path)
