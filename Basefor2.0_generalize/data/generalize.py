"""Annotation and feature loading for MIntRec2.0 -> MIntRec."""
import csv
import logging
import os
import pickle
from collections import Counter

import numpy as np


def read_annotations(path, dataset, labels, binary_maps=None, filter_unknown=False):
    label_map = {label: i for i, label in enumerate(labels)}
    indexes, label_ids, texts = [], [], []
    skipped = Counter()
    seen = set()
    with open(path, encoding='utf-8-sig', newline='') as stream:
        reader = csv.DictReader(stream, delimiter='\t')
        required = (['season', 'episode', 'clip', 'text', 'label']
                    if dataset == 'MIntRec' else
                    ['Dialogue_id', 'Utterance_id', 'Text', 'Label'])
        if not set(required).issubset(reader.fieldnames or []):
            raise ValueError(f'{path}: expected {dataset} columns {required}')
        for line_number, row in enumerate(reader, 2):
            if any(row.get(key) is None for key in required):
                raise ValueError(f'{path}:{line_number}: incomplete TSV row')
            if dataset == 'MIntRec':
                index = '_'.join(row[key] for key in ('season', 'episode', 'clip'))
                text, label = row['text'], row['label']
            else:
                index = f"dia{row['Dialogue_id']}_utt{row['Utterance_id']}"
                text, label = row['Text'], row['Label']
            mapped = binary_maps.get(label) if binary_maps is not None else label
            if mapped not in label_map:
                if not filter_unknown:
                    raise ValueError(f'{path}:{line_number}: unsupported label {label!r}')
                skipped[label] += 1
                continue
            if index in seen:
                raise ValueError(f'{path}:{line_number}: duplicate sample ID {index}')
            seen.add(index)
            indexes.append(index)
            texts.append(text)
            label_ids.append(label_map[mapped])
    if not indexes:
        raise ValueError(f'{path}: no samples remain after label filtering')
    return indexes, label_ids, texts, dict(skipped)


def as_sequence(feature, context):
    feature = np.asarray(feature, dtype=np.float32)
    # Source video commonly has (T, 1, D); target video may already have (T, D).
    if feature.ndim == 3 and feature.shape[1] == 1:
        feature = feature[:, 0, :]
    if feature.ndim != 2 or min(feature.shape) <= 0:
        raise ValueError(f'{context}: expected nonempty (T,D) or (T,1,D), got {feature.shape}')
    if not np.isfinite(feature).all():
        raise ValueError(f'{context}: feature contains NaN or infinity')
    return feature


def pad_sequence(feature, length, width, mode='zero', location='end'):
    if mode not in ('zero', 'normal') or location not in ('start', 'end'):
        raise ValueError('Unsupported padding mode or location')
    if length <= 0 or width < feature.shape[1]:
        raise ValueError('Invalid output length or feature width')
    feature = feature[:length]
    result = np.zeros((length, width), dtype=np.float32)
    offset = length - len(feature) if location == 'start' else 0
    if mode == 'normal' and len(feature) < length:
        result[:, :feature.shape[1]] = np.random.normal(
            feature.mean(), feature.std(), (length, feature.shape[1]))
    result[offset:offset + len(feature), :feature.shape[1]] = feature
    return result


def load_modality(args, attrs, modality):
    logger = logging.getLogger(args.logger_name)
    folder = getattr(args, f'{modality}_data_path')
    source_path = os.path.join(attrs['data_path'], folder, getattr(args, f'{modality}_feats_path'))
    target_path = os.path.join(attrs['data_path'], folder, getattr(args, f'test_{modality}_feats_path'))
    feats, widths = {}, {}
    # Load each pickle once, and release unrelated entries before loading the other domain.
    for path, splits in ((source_path, ('train', 'dev')), (target_path, ('test',))):
        with open(path, 'rb') as stream:
            mapping = pickle.load(stream)
        for split in splits:
            feats[split] = []
            for index in attrs[f'{split}_data_index']:
                if index not in mapping:
                    raise KeyError(f'{path}: missing {split} sample {index}')
                feats[split].append(as_sequence(mapping[index], f'{path}:{index}'))
            split_widths = {feature.shape[1] for feature in feats[split]}
            if len(split_widths) != 1:
                raise ValueError(f'{path}: inconsistent {split} feature widths {split_widths}')
            widths[split] = split_widths.pop()
        del mapping
    if widths['train'] != widths['dev']:
        raise ValueError(f'{modality}: train and dev feature widths must match')
    width = max(widths.values())
    setattr(args, f'{modality}_feat_dim', width)
    attrs['benchmarks']['feat_dims'][modality] = width
    length = getattr(args, f'{modality}_seq_len')
    logger.info('%s feature widths=%s; unified shape=(%d, %d); source=%s; test=%s',
                modality, widths, length, width, source_path, target_path)
    if widths['train'] != widths['test']:
        raise ValueError(
            f'{modality}: source/test feature widths differ ({widths["train"]} vs '
            f'{widths["test"]}); extract both datasets with the same feature encoder')
    attrs['generalization_report'][modality] = {
        'source_path': source_path, 'test_path': target_path,
        'widths': widths, 'output_width': width, 'sequence_length': length,
        'truncated_samples': {split: sum(len(f) > length for f in values)
                              for split, values in feats.items()},
    }
    return {split: [pad_sequence(f, length, width, args.padding_mode, args.padding_loc)
                    for f in values] for split, values in feats.items()}
