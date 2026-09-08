import importlib
from types import SimpleNamespace
from easydict import EasyDict

class ParamManager:
    
    def __init__(self, args):

        # hyper_param, common_param , cmg_cfg= self._get_config_param(args)
        hyper_param, common_param = self._get_config_param(args)

        self.args = EasyDict(
                                dict(
                                        vars(args),
                                        **common_param,
                                        **hyper_param,
                                        # cmg_cfg=SimpleNamespace(**cmg_cfg)   # ✅ 单独挂一个命名空间
                                     )
                            )

    def _get_config_param(self, args):
        
        if args.config_file_name.endswith('.py'):
            module_name = '.' + args.config_file_name[:-3]
        else:
            module_name = '.' + args.config_file_name

        config = importlib.import_module(module_name, 'configs')

        config_param = config.Param
        method_args = config_param(args)

        return method_args.hyper_param, method_args.common_param