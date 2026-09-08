from .generalize import load_modality

__all__ = ['AudioDataset']


class AudioDataset:
    def __init__(self, args, base_attrs):
        self.feats = load_modality(args, base_attrs, 'audio')
