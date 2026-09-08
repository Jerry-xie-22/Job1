class Param():
    
    def __init__(self, args):
        
        self.common_param = self._get_common_parameters(args)
        self.hyper_param = self._get_hyper_parameters(args)
        self.cmg_cfg = self._get_cmg_cfg(args)   # 新增
        # self.self_super_cfg = self._get_self_super_cfg(args)

    def _get_common_parameters(self, args):
        """
            padding_mode (str): The mode for sequence padding ('zero' or 'normal').
            padding_loc (str): The location for sequence padding ('start' or 'end'). 
            eval_monitor (str): The monitor for evaluation ('loss' or metrics, e.g., 'f1', 'acc', 'precision', 'recall').  
            need_aligned: (bool): Whether to perform data alignment between different modalities.
            train_batch_size (int): The batch size for training.
            eval_batch_size (int): The batch size for evaluation. 
            test_batch_size (int): The batch size for testing.
            wait_patience (int): Patient steps for Early Stop.
        """
        common_parameters = {
            'padding_mode': 'zero',
            'padding_loc': 'end',
            'need_aligned': True,
            'eval_monitor': 'f1',
            'train_batch_size': 16,
            'eval_batch_size': 8,
            'test_batch_size': 8,
            'wait_patience': 4
        }
        return common_parameters

    def _get_hyper_parameters(self, args):
        """
        Args:
            num_train_epochs (int): The number of training epochs.
            beta_shift (float): The coefficient for nonverbal displacement to create the multimodal vector.
            dropout_prob (float): The embedding dropout probability.
            warmup_proportion (float): The warmup ratio for learning rate.
            lr (float): The learning rate of backbone.
            aligned_method (str): The method for aligning different modalities. ('ctc', 'conv1d', 'avg_pool')
            weight_decay (float): The coefficient for L2 regularization. 
        """
        hyper_parameters = {
            'num_train_epochs': 100,
            'beta_shift': 0.005,
            # 'dropout_prob': [0.6,0.5,0.45,0.4,0.3],
            'dropout_prob': [0.55],
            'warmup_proportion': 0.1,
            # 'lr': [2e-5,0.000025,3e-5,1e-5],
            # 'lr': [9e-06,5e-06,7e-06,1.5e-05,3.5e-05,4e-05,5e-05],
            'lr': [2e-05],
            'aligned_method': 'ctc',
            'weight_decay': [0.03],
            'num_experts':[2],
            'nheads': [8], 
            'n_levels': [5], 
            'attn_dropout': [0.1], 
            'relu_dropout': 0.0, 
            'embed_dropout': [0.2], 
            'res_dropout': 0.1,
            'attn_mask': True,
        }
        return hyper_parameters 
    
    def _get_cmg_cfg(self, args):
        return {
            "bidirectional": False,
            "attn_dropout_a": 0.2,
            "attn_dropout_v": 0.0,
            "relu_dropout": 0.0,
            "embed_dropout": 0.2,
            "res_dropout": 0.0,
            "dst_feature_dim_nheads": 128,
            # "batch_size": 32,
            # "learning_rate": 0.002,
            "nlevels": 4,
            "conv1d_kernel_size_l": 5,
            "conv1d_kernel_size_a": 5,
            "conv1d_kernel_size_v": 5,
            "text_dropout": 0.5,
            "attn_dropout": 0.3,
            "output_dropout": 0.5,
            "grad_clip": 0.6,
            "patience": 5,
            "weight_decay": 0.005,
            "use_xlstm": True,
            "text_out": 768,
            "audio_embed": 768,
            "audio_feature_dim": 16,
            "video_feature_dim": 32,
            "audio_out": 128,
            "resnet_out": 512,
            "video_out": 128,
            "attn_mask": False
        }
    
    # def _get_self_super_cfg(self, args):
    #     return {
    #         "text_embed": 768,
    #         "v_res_in": 512,
    #         "a_d2v_in": 768,
    #         "v_res_seq_len": 15,
    #         "a_d2v_seq_len": 46,

    #         # "v_feat_in": 20,
    #         # "a_feat_in": 5,
    #         # "v_feat_out": 128,
    #         # "a_feat_out": 128,
    #         "v_feat_seq_len": 500,
    #         "a_feat_seq_len": 375,

    #         "inter_num_heads": 1,
    #         "inter_seq_lens": [50, 375, 500],
    #         "inter_pos_dropout": 0.1,
    #         "inter_random_masking_rate": 0.1,
    #         "num_inter_graph_layers": 1,

    #         "intra_num_heads": 1,
    #         "intra_seq_lens": [50, 15, 46],
    #         "intra_pos_dropout": 0.1,
    #         "intra_random_masking_rate": 0.1,
    #         "num_intra_graph_layers": 1,

    #         "beta_shift": 1,
    #         "shifting_gate_dropout_prob": 0.3,

    #         "inter_out": 64,
    #         "intra_out": 128
    #     }