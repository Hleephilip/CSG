import torch
from diffusers.models.attention import CrossAttention

class MyCrossAttnProcessor:
    def __call__(self, attn: CrossAttention, hidden_states, encoder_hidden_states=None, attention_mask=None):
        if attn.inject_attn == 0:
            _, sequence_length, _ = hidden_states.shape
            attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length)

            query = attn.to_q(hidden_states)

            encoder_hidden_states = encoder_hidden_states if encoder_hidden_states is not None else hidden_states
            key = attn.to_k(encoder_hidden_states)
            value = attn.to_v(encoder_hidden_states)

            query = attn.head_to_batch_dim(query)
            key = attn.head_to_batch_dim(key)
            value = attn.head_to_batch_dim(value)

            attention_probs = attn.get_attention_scores(query, key, attention_mask)
            # new bookkeeping to save the attn probs
            attn.attn_probs = attention_probs

            hidden_states = torch.bmm(attention_probs, value)
            hidden_states = attn.batch_to_head_dim(hidden_states)

            # linear proj
            hidden_states = attn.to_out[0](hidden_states)
            # dropout
            hidden_states = attn.to_out[1](hidden_states)

            return hidden_states
        
        elif attn.inject_attn == 1:
            encoder_hidden_states = encoder_hidden_states if encoder_hidden_states is not None else hidden_states
            value = attn.to_v(encoder_hidden_states)
            value = attn.head_to_batch_dim(value)
            hidden_states = torch.bmm(attn.attn_probs, value) # use injected attention map
            hidden_states = attn.batch_to_head_dim(hidden_states)

            # linear proj
            hidden_states = attn.to_out[0](hidden_states)
            # dropout
            hidden_states = attn.to_out[1](hidden_states)
            attn.inject_attn = 0
            return hidden_states


class MySelfAttnProcessor:
    def __call__(self, attn: CrossAttention, hidden_states, encoder_hidden_states=None, attention_mask=None):
        _, sequence_length, _ = hidden_states.shape
        attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length)

        query = attn.to_q(hidden_states)

        encoder_hidden_states = encoder_hidden_states if encoder_hidden_states is not None else hidden_states
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        query = attn.head_to_batch_dim(query)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)

        attention_probs = attn.get_attention_scores(query, key, attention_mask)
        # new bookkeeping to save the attn probs
        attn.attn_probs = attention_probs

        hidden_states = torch.bmm(attention_probs, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        return hidden_states

"""
A function that prepares a U-Net model for training by enabling gradient computation
for a specified set of parameters and setting the forward pass to be performed by a
custom cross attention processor.

Parameters:
unet: A U-Net model.

Returns:
unet: The prepared U-Net model.
"""
def prep_unet_my_cross_self(unet):
    # replace the fwd function
    for module in unet.named_children():
        if 'down' in module[0] or 'up' in module[0] or 'mid' in module[0]:
            for name, sub_module in module[1].named_modules():
                sub_module_name = type(sub_module).__name__
                # if sub_module_name == "CrossAttention":
                if sub_module_name == "CrossAttention" and 'attn2' in name:
                    sub_module.inject_attn = 0
                    sub_module.set_processor(MyCrossAttnProcessor())
    unet.up_blocks[-1].attentions[-1].transformer_blocks[0].attn1.set_processor(MySelfAttnProcessor())
    return unet
