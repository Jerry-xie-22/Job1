from .mag_bert import Param as BaseParam


class Param(BaseParam):
    """Run the full control and three ablations with fixed Base hyperparameters."""

    def _get_hyper_parameters(self, args):
        parameters = super()._get_hyper_parameters(args)
        parameters['ablation_mode'] = [
            'full',
            'label_cons_only',
            'label_classifier_only',
            'without_both',
        ]
        return parameters
