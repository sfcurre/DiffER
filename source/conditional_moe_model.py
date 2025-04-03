
import torch
import torch.nn as nn
from .conditional_model import ConditionalModel
from switch_transformers import SwitchMoE

'''
This code uses monkeypatching to adapt pytorch decoders into MoE models.
'''

class ConditionalMoEModel(ConditionalModel):
    def __init__(self, *args, num_experts=4, return_aux_loss=False, **kwargs):
        super(ConditionalMoEModel, self).__init__(*args, **kwargs)
        self.num_experts = num_experts
        self.return_aux_loss = return_aux_loss
        self.moes = nn.ModuleList([self.build_moe() for i in range(self.num_layers)])
        self.patch_feedforward()
        self.aux_loss = 0

    def patch_feedforward_layer(self, m, moe):
        m.linear1=None
        m.linear2=None
        
        def moe_ff_block(x):
            output, loss = moe(x)
            if self.return_aux_loss:
                self.aux_loss += loss
            return output
        
        m._ff_block = moe_ff_block
    
    def build_moe(self):
        moe = SwitchMoE(dim=self.d_model, 
                        hidden_dim=self.d_feedforward, 
                        output_dim=self.d_model, 
                        num_experts=self.num_experts, 
                        mult=self.d_feedforward // self.d_model,
                        dropout=self.dropout_,
                        use_aux_loss=self.return_aux_loss)
        return moe
        
    def patch_feedforward(self):
        for m, moe in zip(self.decoder.layers, self.moes):
            self.patch_feedforward_layer(m, moe)

    def forward(self, batch):
        memory, memory_pad_mask, predicted_lengths = self.encode(batch['y_0'], batch['y_mask'])
        logits = self.decode(batch['x_t'], batch['x_mask'], memory, memory_pad_mask, batch['t'])
        if self.return_aux_loss:
            aux_loss = self.aux_loss
            self.aux_loss = 0
            return logits, predicted_lengths, aux_loss
        return logits, predicted_lengths
    
    def get_aux_loss(self):
        aux_loss = self.aux_loss
        self.aux_loss = 0
        return aux_loss
