from transformers.configuration_utils import PretrainedConfig

class TeKANConfig(PretrainedConfig):
    model_type = "tekan"

    def __init__(
        self,
        pad_token_id=1,
        bos_token_id=0,
        eos_token_id=2,
        num_classes=2,
        hidden_size=768,
        num_channels=[256, 256, 256],
        kernel_sizes=[3, 4, 5],
        dropout=0.1,
        **kwargs
    ):
        super().__init__(pad_token_id=pad_token_id, bos_token_id=bos_token_id, eos_token_id=eos_token_id, **kwargs)

        self.num_classes = num_classes
        self.hidden_size = hidden_size
        self.num_channels = num_channels
        self.kernel_sizes = kernel_sizes
        self.dropout = dropout