import csv
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
        self.labels = benchmarks['MIntRec']['intent_labels']

    def write_tsv(self, name, rows):
        path = self.root / name
        with path.open('w', encoding='utf-8', newline='') as stream:
            csv.writer(stream, delimiter='\t').writerows(rows)
        return path

    def test_target_labels_and_ids_preserve_row_alignment(self):
        path = self.write_tsv('test.tsv', [
            ['Dialogue_id', 'Utterance_id', 'Text', 'Label'],
            ['9', '1', 'first', 'Flaunt'], ['9', '2', 'excluded', 'Confirm'],
            ['9', '3', 'third', 'Inform'], ['9', '4', 'unknown', 'UNK']])
        ids, labels, texts, skipped = read_annotations(
            path, 'MIntRec2.0', self.labels, filter_unknown=True)
        self.assertEqual(ids, ['dia9_utt1', 'dia9_utt3'])
        self.assertEqual(texts, ['first', 'third'])
        self.assertEqual(labels, [self.labels.index('Flaunt'), self.labels.index('Inform')])
        self.assertEqual(skipped, {'Confirm': 1, 'UNK': 1})

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
        target = {'same': np.full((1, 2), 7)}
        for name, values in ((modality + '_feats.pkl', source),
                             (modality + '_feats2.pkl', target)):
            with (folder / name).open('wb') as stream:
                pickle.dump(values, stream)
        args = SimpleNamespace(logger_name='test', padding_mode='zero', padding_loc='end')
        for suffix, value in (('data_path', folder.name), ('feats_path', modality + '_feats.pkl'),
                              ('seq_len', 3)):
            setattr(args, modality + '_' + suffix, value)
        setattr(args, 'test_' + modality + '_feats_path', modality + '_feats2.pkl')
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
            np.testing.assert_array_equal(feats['test'][0][0], [7, 7, 0, 0])
            np.testing.assert_array_equal(feats['train'][0][0], [1, 1, 1, 1])
            attrs['test_data_index'] = ['missing']
            with self.assertRaisesRegex(KeyError, 'missing test sample'):
                load_modality(args, attrs, modality)

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
        for split in ('train', 'dev'):
            ids, labels, texts, skipped = read_annotations(
                samples / 'MIntRec' / (split + '.tsv'), 'MIntRec', self.labels)
            self.assertFalse(skipped)
            self.assertEqual(len(ids), len(texts))
        ids, labels, texts, skipped = read_annotations(
            samples / 'MIntRec2.0/test.tsv', 'MIntRec2.0', self.labels, filter_unknown=True)
        self.assertEqual(len(ids), len(texts))
        self.assertTrue(all(0 <= label < 20 for label in labels))
        self.assertTrue(skipped)
        print(f'Target TSV: retained={len(ids)}, excluded={sum(skipped.values())}, labels={skipped}')


if __name__ == '__main__':
    unittest.main()
