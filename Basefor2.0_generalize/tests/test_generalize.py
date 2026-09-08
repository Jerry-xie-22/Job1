import csv
import json
import importlib.util
import pickle
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from data import benchmarks
from data.generalize import as_sequence, load_modality, pad_sequence, read_annotations


class GeneralizationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.labels = benchmarks['MIntRec2.0']['intent_labels']

    def write_tsv(self, name, rows):
        path = self.root / name
        with path.open('w', encoding='utf-8', newline='') as stream:
            csv.writer(stream, delimiter='\t').writerows(rows)
        return path

    def test_target_labels_and_ids_preserve_row_alignment(self):
        path = self.write_tsv('test.tsv', [
            ['Dialogue_id', 'Utterance_id', 'Text', 'Label'],
            ['9', '1', 'first', 'Flaunt'], ['9', '2', 'excluded', 'NotAnIntent'],
            ['9', '3', 'third', 'Inform'], ['9', '4', 'unknown', 'UNK']])
        ids, labels, texts, skipped = read_annotations(
            path, 'MIntRec2.0', self.labels, filter_unknown=True)
        self.assertEqual(ids, ['dia9_utt1', 'dia9_utt3'])
        self.assertEqual(texts, ['first', 'third'])
        self.assertEqual(labels, [self.labels.index('Flaunt'), self.labels.index('Inform')])
        self.assertEqual(skipped, {'NotAnIntent': 1, 'UNK': 1})

    def test_source_and_invalid_labels(self):
        path = self.write_tsv('train.tsv', [
            ['season', 'episode', 'clip', 'text', 'label'],
            ['S01', 'E02', '3', 'hello', 'Greet']])
        ids, labels, texts, skipped = read_annotations(path, 'MIntRec', self.labels)
        self.assertEqual(ids, ['S01_E02_3'])
        self.assertEqual(texts, ['hello'])
        with self.assertRaises(ValueError):
            read_annotations(path, 'MIntRec2.0', self.labels)
        with self.assertRaises(ValueError):
            read_annotations(path, 'MIntRec', ['Inform'])

    def test_shapes_padding_and_truncation(self):
        source = as_sequence(np.ones((5, 1, 4)), 'source')
        target = as_sequence(np.ones((1, 2)), 'target')
        self.assertEqual(source.shape, (5, 4))
        out = pad_sequence(target, 3, 4)
        np.testing.assert_array_equal(out[0], [1, 1, 0, 0])
        np.testing.assert_array_equal(out[1:], 0)
        self.assertEqual(out.dtype, np.float32)
        np.testing.assert_array_equal(pad_sequence(source, 2, 4), np.ones((2, 4)))
        np.testing.assert_array_equal(pad_sequence(target, 3, 4, location='start')[-1], [1, 1, 0, 0])
        np.testing.assert_array_equal(pad_sequence(target, 3, 4, mode='normal')[:, 2:], 0)
        for bad in (np.ones((2, 2, 3)), np.ones((0, 4)), np.array([[np.nan]])):
            with self.assertRaises(ValueError):
                as_sequence(bad, 'bad')

    def feature_fixture(self, modality):
        folder = self.root / (modality + '_data')
        folder.mkdir()
        # Same ID in different domains: test must select the target pickle.
        source = {'same': np.full((2, 1, 4), 1), 'dev': np.full((1, 4), 2)}
        target = {'same': np.full((1, 4), 7)}
        for name, values in ((modality + '_feats.pkl', source),
                             (modality + '_feats1.pkl', target)):
            with (folder / name).open('wb') as stream:
                pickle.dump(values, stream)
        args = SimpleNamespace(logger_name='test', padding_mode='zero', padding_loc='end')
        for suffix, value in (('data_path', folder.name), ('feats_path', modality + '_feats.pkl'),
                              ('seq_len', 3)):
            setattr(args, modality + '_' + suffix, value)
        setattr(args, 'test_' + modality + '_feats_path', modality + '_feats1.pkl')
        attrs = {'data_path': str(self.root), 'train_data_index': ['same'],
                 'dev_data_index': ['dev'], 'test_data_index': ['same'],
                 'benchmarks': {'feat_dims': {}}, 'generalization_report': {}}
        return args, attrs

    def test_audio_and_video_use_separate_pickles_and_matching_widths(self):
        for modality in ('audio', 'video'):
            args, attrs = self.feature_fixture(modality)
            feats = load_modality(args, attrs, modality)
            self.assertEqual(getattr(args, modality + '_feat_dim'), 4)
            self.assertEqual(feats['test'][0].shape, (3, 4))
            np.testing.assert_array_equal(feats['test'][0][0], [7, 7, 7, 7])
            np.testing.assert_array_equal(feats['train'][0][0], [1, 1, 1, 1])
            attrs['test_data_index'] = ['missing']
            with self.assertRaisesRegex(KeyError, 'missing test sample'):
                load_modality(args, attrs, modality)

    def test_mismatched_domain_widths_are_rejected(self):
        args, attrs = self.feature_fixture('video')
        path = self.root / 'video_data/video_feats1.pkl'
        with path.open('wb') as stream:
            pickle.dump({'same': np.ones((1, 2))}, stream)
        with self.assertRaisesRegex(ValueError, 'source/test feature widths differ'):
            load_modality(args, attrs, 'video')

    def test_text_uses_filtered_annotations_not_raw_tsv(self):
        # No local transformers/BERT weights needed: stub only the tokenizer boundary.
        fake = SimpleNamespace(BertTokenizer=object)
        with patch.dict(sys.modules, {'transformers': fake}):
            spec = importlib.util.spec_from_file_location('generalize_text_test', ROOT / 'data/text_pre.py')
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
        attrs = {}
        for split in ('train', 'dev', 'test'):
            attrs[split + '_data_index'] = ['id-' + split]
            attrs[split + '_data_text'] = ['text-' + split]
        args = SimpleNamespace(logger_name='test', text_backbone='bert-large-uncased')
        with patch.object(module.TextDataset, '_get_bert_feats',
                          side_effect=lambda args, examples, attrs: [(e.guid, e.text_a) for e in examples]):
            feats = module.TextDataset(args, attrs).feats
        self.assertEqual(feats['test'], [('id-test', 'text-test')])

    def test_supplied_tsvs(self):
        samples = ROOT.parent / 'MInteRec_data'
        if not samples.exists():
            self.skipTest('Local TSV reference files are not part of the code repository')
        expected_counts = {'train': 4125, 'dev': 726}
        for split in ('train', 'dev'):
            source = samples / 'MIntRec2.0' / (split + '.tsv')
            bundled = ROOT / f'MIntRec2.0_{split}_20.tsv'
            expected = read_annotations(source, 'MIntRec2.0', self.labels, filter_unknown=True)
            actual = read_annotations(bundled, 'MIntRec2.0', self.labels)
            self.assertEqual(actual[:3], expected[:3])
            self.assertEqual(len(actual[0]), expected_counts[split])
            self.assertEqual(sum(expected[3].values()),
                             {'train': 2040, 'dev': 380}[split])
        ids, labels, texts, skipped = read_annotations(
            samples / 'MIntRec/test.tsv', 'MIntRec', self.labels)
        self.assertEqual(len(ids), len(texts))
        self.assertTrue(all(0 <= label < 20 for label in labels))
        self.assertFalse(skipped)
        with (samples / 'MIntRec/test.tsv').open(encoding='utf-8-sig', newline='') as stream:
            raw_rows = list(csv.DictReader(stream, delimiter='\t'))
        self.assertEqual(len(ids), len(raw_rows))
        self.assertEqual(labels[0], self.labels.index('Arrange'))
        self.assertEqual(labels[0], benchmarks['MIntRec']['intent_labels'].index('Arrange'))
        print(f'Target TSV: retained={len(ids)}, excluded={sum(skipped.values())}, labels={skipped}')

    def test_data_manager_end_to_end_with_tensor_and_tokenizer_stubs(self):
        # Exercise production TSV, pickle, padding, dataset and report wiring.
        # Only replace unavailable PyTorch/Transformers boundaries, never the loaders.
        from importlib import import_module
        dataset_dir = self.root / 'MIntRec2.0'
        dataset_dir.mkdir()
        self.root = dataset_dir
        source_header = ['Dialogue_id', 'Utterance_id', 'Text', 'Label']
        train_path = self.write_tsv('train.tsv', [source_header, ['1', '1', 'source train', 'Inform']])
        dev_path = self.write_tsv('dev.tsv', [source_header, ['2', '1', 'source dev', 'Greet']])
        self.write_tsv('test.tsv', [['season', 'episode', 'clip', 'text', 'label'],
                                  ['S01', 'E01', '1', 'target text', 'Arrange']])
        for modality in ('audio', 'video'):
            folder = dataset_dir / (modality + '_data')
            folder.mkdir()
            for suffix, mapping in [('', {'dia1_utt1': np.ones((2, 1, 4)),
                                         'dia2_utt1': np.ones((1, 4))}),
                                    ('1', {'S01_E01_1': np.full((1, 4), 7)})]:
                with (folder / (modality + '_feats' + suffix + '.pkl')).open('wb') as f:
                    pickle.dump(mapping, f)
        class Tokenizer:
            @classmethod
            def from_pretrained(cls, *args, **kwargs):
                return cls()
            def tokenize(self, text):
                return text.split()
            def convert_tokens_to_ids(self, tokens):
                return list(range(len(tokens)))
        fake_data = SimpleNamespace(Dataset=object,
                                    DataLoader=lambda dataset, **kwargs: SimpleNamespace(dataset=dataset, **kwargs))
        fake_modules = {'torch': SimpleNamespace(tensor=np.asarray),
                        'torch.utils': SimpleNamespace(data=fake_data),
                        'torch.utils.data': fake_data,
                        'transformers': SimpleNamespace(BertTokenizer=Tokenizer)}
        args = SimpleNamespace(
            logger_name='integration', dataset='MIntRec2.0', data_mode='multi-class',
            data_path=str(dataset_dir.parent), text_backbone='bert-large-uncased',
            audio_data_path='audio_data', video_data_path='video_data',
            audio_feats_path='audio_feats.pkl', video_feats_path='video_feats.pkl',
            test_audio_feats_path='audio_feats1.pkl', test_video_feats_path='video_feats1.pkl',
            train_tsv_path=str(train_path), dev_tsv_path=str(dev_path),
            padding_mode='zero', padding_loc='end', train_batch_size=2, eval_batch_size=3,
            test_batch_size=4, num_workers=0, results_path=str(dataset_dir / 'reports'))
        original_width = benchmarks['MIntRec2.0']['feat_dims']['video']
        with patch.dict(sys.modules, fake_modules):
            manager = import_module('data.base').DataManager(args)
        self.assertEqual(args.num_labels, 20)
        self.assertEqual((args.text_seq_len, args.video_seq_len, args.audio_seq_len), (76, 230, 480))
        self.assertEqual((args.video_feat_dim, args.audio_feat_dim), (4, 4))
        sample = manager.mm_data['test'][0]
        self.assertEqual(sample['text'], 'target text')
        self.assertEqual(sample['sample_index'], 'S01_E01_1')
        self.assertEqual(sample['label_ids'], self.labels.index('Arrange'))
        self.assertEqual(sample['text_feats'].shape, (3, 76))
        np.testing.assert_array_equal(sample['video_feats'][0], [7, 7, 7, 7])
        self.assertTrue(manager.mm_dataloader['train'].shuffle)
        self.assertFalse(manager.mm_dataloader['test'].shuffle)
        self.assertEqual(manager.mm_dataloader['test'].batch_size, 4)
        report = json.loads((dataset_dir / 'reports/integration_data_report.json').read_text())
        self.assertEqual(report['splits']['test']['coverage'], 1)
        self.assertEqual(report['source_dataset'], 'MIntRec2.0')
        self.assertEqual(benchmarks['MIntRec2.0']['feat_dims']['video'], original_width)

    def test_shared_label_order_matches_mintrec(self):
        self.assertEqual(benchmarks['MIntRec2.0']['intent_labels'],
                         benchmarks['MIntRec']['intent_labels'])
        self.assertEqual(len(self.labels), 20)


if __name__ == '__main__':
    unittest.main()
