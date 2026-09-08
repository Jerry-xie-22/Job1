from .generalize import load_modality

__all__ = ['VideoDataset']


class VideoDataset:
    def __init__(self, args, base_attrs):
        self.feats = load_modality(args, base_attrs, 'video')
