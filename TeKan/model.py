from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor
from transformers import PreTrainedModel, AutoModel

from .configuration import TeKANConfig
from .kan import KANLinear
from .conv import KanConv1D

from functools import reduce

class TeKANPretrainModel(PreTrainedModel):
    config_class = TeKANConfig
    base_model_prefix = "tekan"
    supports_gradient_checkpointing = True

    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    @property
    def dummy_inputs(self):
        pad_token = self.config.pad_token_id
        input_ids = torch.tensor([[0, 6, 10, 4, 2], [0, 8, 12, 2, pad_token]], device=self.device)
        dummy_inputs = {
            "attention_mask": input_ids.ne(pad_token),
            "input_ids": input_ids,
            "decoder_input_ids": input_ids,
        }
        return dummy_inputs
    
class TeKAN(TeKANPretrainModel):
    _keys_to_ignore_on_load_unexpected = [r"pooler"]
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def __init__(self, config: TeKANConfig):
        super().__init__(config)

        self.config = config

        self.conv1d_layer = nn.ModuleList()
        for kernel_size, channel in zip(self.config.kernel_sizes, self.config.num_channels):
            self.conv1d_layer.append(
                KanConv1D(self.config.hidden_size, channel, kernel_size)
            )

        # Pooling
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.relu = nn.ReLU()

        # projection to classification
        hidden_dim = reduce(lambda x, y: x + y, self.config.num_channels)
        self.proj = KANLinear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(self.config.dropout)
        self.tanh = nn.Tanh()
        self.out = KANLinear(hidden_dim, self.config.num_classes)

    def forward(
        self,
        hidden_states: Optional[torch.FloatTensor] = None,
    ) -> Tensor:
        # hidden_states shape (B, S, E)
        we_output = hidden_states.permute(0, 2, 1) # (B, E, S)

        # after layer->pool->relu -> (B, Channel, 1)
        # squeeze -> (B, Channel)
        conv1d_res = []
        for layer in self.conv1d_layer:
            conv1d_res.append(
                torch.squeeze(self.relu(self.pool(layer(we_output))), dim=-1)
            )
        merge = torch.cat(conv1d_res, dim=1)
        output = self.tanh(self.dropout(self.proj(merge)))
        cls_output = self.out(output)
        return cls_output
    
class TeKANClassifierModel(nn.Module):
    def __init__(self, pretrained_model: AutoModel, classifier_model: TeKAN):
        self.pretrained_model = pretrained_model
        self.classifier_model = classifier_model

        for p in self.pretrained_model.parameters():
            p.requires_grad = False

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Tensor:
        with torch.inference_mode():
            embeds = self.pretrained_model(
                input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,)
        output = self.classifier_model(embeds)
        return output
    
    def save_pretrained(self, path):
        self.classifier_model.save_pretrained(path)