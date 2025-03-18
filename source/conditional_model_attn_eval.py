
import torch
from .conditional_model import ConditionalModel

'''
This code is adapted from https://gist.github.com/airalcorn2/50ec06517ce96ecc143503e21fa6cb91
'''

class ConditionalModelAttnEval(ConditionalModel):
    def __init__(self, *args, **kwargs):
        super(ConditionalModelAttnEval, self).__init__(*args, **kwargs)
        self.encoder_attn_outputs = {}
        self.decoder_attn_outputs = {}
        self.patch_attention()

    def patch_attention_layer(self, m):
        forward_orig = m.forward

        def wrap(*args, **kwargs):
            kwargs["need_weights"] = True
            kwargs["average_attn_weights"] = False

            return forward_orig(*args, **kwargs)

        m.forward = wrap

    def patch_attention(self):
        for m in self.encoder.layers:
            self.patch_attention_layer(m.self_attn)
            m.register_forward_hook(self.save_output)

        for m in self.decoder.layers:
            self.patch_attention_layer(m.self_attn)
            m.register_forward_hook(self.save_output)

    def save_output_encoder(self, m, i, o):
        self.encoder_attn_outputs[m] = o

    def save_output_decoder(self, m, i, o):
        self.decoder_attn_outputs[m] = o
    
    def get_attn(self, mode='encoder'):
        if mode == 'encoder':
            return self.encoder_attn_outputs.copy()
        elif mode == 'decoder':
            return self.decoder_attn_outputs.copy()
        else: 
            return self.encoder_attn_outputs.copy(), self.decoder_attn_outputs.copy()