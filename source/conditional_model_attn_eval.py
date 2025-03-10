
import torch
from .conditional_model import ConditionalModel

'''
This code is adapted from https://gist.github.com/airalcorn2/50ec06517ce96ecc143503e21fa6cb91
'''

class ConditionalModelAttnEval(ConditionalModel):
    def __init__(self, *args, **kwargs):
        super(ConditionalModelAttnEval, self).__init__(*args, **kwargs)
        self.attn_outputs = {}
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

    def save_output(self, m, i, o):
        self.attn_outputs[m] = o

    def export(self, path):
        torch.save(self.attn_outputs, path)