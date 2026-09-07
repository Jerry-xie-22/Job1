import torch
import logging
from torch import nn
from .__init__ import methods_map

__all__ = ['ModelManager']

# 创建一个对三种特征进行融合分析的类，其实就是MAG_BERT, MISA,MULT三个类
class MIA(nn.Module): 

    def __init__(self, args):

        super(MIA, self).__init__()

        # 根据融合方法，选择对应的类
        fusion_method = methods_map[args.method]
        self.model = fusion_method(args)

    def forward(self, text_feats, video_feats, audio_feats):
        video_feats, audio_feats = video_feats.float(), audio_feats.float()
        # 将文本特征、视频特征和音频特征传入模型（MAG_BERT, MISA,MULT）进行融合，self.model就是MAG_BERT, MISA,MULT种的一个
        mm_model = self.model(text_feats, video_feats, audio_feats)

        return mm_model
        
class ModelManager:

    def __init__(self, args):
        
        self.logger = logging.getLogger(args.logger_name)
        # 获取设备，如果有GPU可用，则使用GPU，否则使用CPU
        self.device = args.device = torch.device('cuda:%d' % int(args.gpu_id) if torch.cuda.is_available() else 'cpu')
        # 设置模型
        self.model = self._set_model(args)

    def _set_model(self, args):

        model = MIA(args) 
        model.to(self.device) # 将模型移动到指定的计算设备上
        return model