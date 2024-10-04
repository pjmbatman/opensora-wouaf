import torch
import torch.nn.functional as F
from dataclasses import dataclass
from diffusers.utils import BaseOutput
from typing import Any, Dict, List, Optional, Tuple, Union
#from diffusers.models.unet_2d_blocks import UNetMidBlock2D, UpDecoderBlock2D, CrossAttnDownBlock2D, DownBlock2D, UNetMidBlock2DCrossAttn, UpBlock2D, CrossAttnUpBlock2D
from diffusers.models.unets.unet_2d_blocks import UNetMidBlock2D, UpDecoderBlock2D
from diffusers.models.resnet import ResnetBlock2D
from diffusers.models.attention import Attention
from attribution import FullyConnectedLayer
import math


def customize_spatial_vae_decoder(vae, phi_dimension, lr_multiplier):
    print(f"size of phi dimension: {phi_dimension}")
    def add_affine_conv(vaed):
        for layer in vaed.children():
            if isinstance(layer, ResnetBlock2D):
                layer.affine1 = FullyConnectedLayer(phi_dimension, layer.conv1.weight.shape[1], lr_multiplier=lr_multiplier, bias_init=1)
                layer.affine2 = FullyConnectedLayer(phi_dimension, layer.conv2.weight.shape[1], lr_multiplier=lr_multiplier, bias_init=1)
            else:
                add_affine_conv(layer)

    def add_affine_attn(vaed):
        for layer in vaed.children():
            if isinstance(layer, Attention):
                layer.affine_q = FullyConnectedLayer(phi_dimension, layer.to_q.weight.shape[1], lr_multiplier=lr_multiplier, bias_init=1)
                layer.affine_k = FullyConnectedLayer(phi_dimension, layer.to_k.weight.shape[1], lr_multiplier=lr_multiplier, bias_init=1)
                layer.affine_v = FullyConnectedLayer(phi_dimension, layer.to_v.weight.shape[1], lr_multiplier=lr_multiplier, bias_init=1)
            else:
                add_affine_attn(layer)
    """ 
    change_forward(vae.decoder, UNetMidBlock2D, new_forward_MB)
    change_forward(vae.decoder, UpDecoderBlock2D, new_forward_UDB)
    change_forward(vae.decoder, ResnetBlock2D, new_forward_RB)
    change_forward(vae.decoder, Attention, new_forward_AB)
    문제: for문 전부 돌아도 UNetMidBlock2D, UpDecoderBlock2D를 못찾음
    해결: UNetMidBlock2D, UpDecoderBlock2D의 구조정의가 달라져서
    from diffusers.models.unet_2d_blocks import UNetMidBlock2D, UpDecoderBlock2D
    ->
    from diffusers.models.unets.unet_2d_blocks import UNetMidBlock2D, UpDecoderBlock2D
    으로 바꿈
    """
    def change_forward(vaed, layer_type, new_forward):
        for layer in vaed.children():
            if isinstance(layer, layer_type):
                #print("성공")
                bound_method = new_forward.__get__(layer, layer.__class__)
                setattr(layer, 'forward', bound_method)
            else:
                # Recursively apply to nested modules
                change_forward(layer, layer_type, new_forward)
    # 수정 및 통과
    def new_forward_MB(self, hidden_states, encoded_fingerprint, temb=None):
        #print(7)
        hidden_states = self.resnets[0]((hidden_states, encoded_fingerprint), temb)
        #print(6)
        for attn, resnet in zip(self.attentions, self.resnets[1:]):
            if attn is not None:
                hidden_states = attn((hidden_states, encoded_fingerprint))
            hidden_states = resnet((hidden_states, encoded_fingerprint), temb)
        #print(hidden_states)
        #print(7)
        return hidden_states
    # 수정 및 통과
    def new_forward_UDB(self, hidden_states, encoded_fingerprint):
        #print(77)
        for resnet in self.resnets:
            hidden_states = resnet((hidden_states, encoded_fingerprint), temb=None)

        if self.upsamplers is not None:
            for upsampler in self.upsamplers:
                hidden_states = upsampler(hidden_states)

        return hidden_states
    # 수정 및 통과
    def new_forward_RB(self, input_tensor, temb):
        input_tensor, encoded_fingerprint = input_tensor
        #print(f"encoded_fingerprint shape: {encoded_fingerprint.shape}")
        #print(f"input_shape : {input_tensor.shape}")
        #print(f"affine1 weight shape: {self.affine1.weight.shape}")
        hidden_states = input_tensor
        #print(hidden_states.size())

        hidden_states = self.norm1(hidden_states)
        hidden_states = self.nonlinearity(hidden_states)

        if self.upsample is not None:
            # upsample_nearest_nhwc fails with large batch sizes. see https://github.com/huggingface/diffusers/issues/984
            if hidden_states.shape[0] >= 64:
                input_tensor = input_tensor.contiguous()
                hidden_states = hidden_states.contiguous()
            input_tensor = self.upsample(input_tensor)
            hidden_states = self.upsample(hidden_states)
        elif self.downsample is not None:
            input_tensor = self.downsample(input_tensor)
            hidden_states = self.downsample(hidden_states)

        phis = self.affine1(encoded_fingerprint)
        batch_size = phis.shape[0]
        #print("라라라라")
        #print(batch_size)
        weight = phis.view(batch_size, 1, -1, 1, 1) * self.conv1.weight.unsqueeze(0)
        hidden_states = F.conv2d(hidden_states.contiguous().view(1, -1, hidden_states.shape[-2], hidden_states.shape[-1]), weight.view(-1, weight.shape[-3], weight.shape[-2], weight.shape[-1]), padding=1, groups=batch_size).view(batch_size, weight.shape[1], hidden_states.shape[-2], hidden_states.shape[-1]) + self.conv1.bias.view(1, -1, 1, 1)

        if temb is not None:
            temb = self.time_emb_proj(self.nonlinearity(temb))[:, :, None, None]

        if temb is not None and self.time_embedding_norm == "default":
            hidden_states = hidden_states + temb

        hidden_states = self.norm2(hidden_states)

        if temb is not None and self.time_embedding_norm == "scale_shift":
            scale, shift = torch.chunk(temb, 2, dim=1)
            hidden_states = hidden_states * (1 + scale) + shift

        hidden_states = self.nonlinearity(hidden_states)

        hidden_states = self.dropout(hidden_states)

        phis = self.affine2(encoded_fingerprint)
        batch_size = phis.shape[0]
        #print("라라라라")
        #print(batch_size)
        weight = phis.view(batch_size, 1, -1, 1, 1) * self.conv2.weight.unsqueeze(0)
        hidden_states = F.conv2d(hidden_states.contiguous().view(1, -1, hidden_states.shape[-2], hidden_states.shape[-1]), weight.view(-1, weight.shape[-3], weight.shape[-2], weight.shape[-1]), padding=1, groups=batch_size).view(batch_size, weight.shape[1], hidden_states.shape[-2], hidden_states.shape[-1]) + self.conv2.bias.view(1, -1, 1, 1)

        if self.conv_shortcut is not None:
            input_tensor = self.conv_shortcut(input_tensor)

        output_tensor = (input_tensor + hidden_states) / self.output_scale_factor

        return output_tensor
    
    # 수정중
    def new_forward_AB(self, hidden_states):
        #print(777)
        hidden_states, encoded_fingerprint = hidden_states
        residual = hidden_states
        batch, channel, height, width = hidden_states.shape

        # norm
        hidden_states = self.group_norm(hidden_states)

        hidden_states = hidden_states.view(batch, channel, height * width).transpose(1, 2)
        
        # proj to q, k, v
        phis_q = self.affine_q(encoded_fingerprint)
        query_proj = torch.bmm(hidden_states, phis_q.unsqueeze(-1) * self.to_q.weight.t().unsqueeze(0)) + self.to_q.bias

        phis_k = self.affine_k(encoded_fingerprint)
        key_proj = torch.bmm(hidden_states, phis_k.unsqueeze(-1) * self.to_k.weight.t().unsqueeze(0)) + self.to_k.bias

        phis_v = self.affine_v(encoded_fingerprint)
        value_proj = torch.bmm(hidden_states, phis_v.unsqueeze(-1) * self.to_v.weight.t().unsqueeze(0)) + self.to_v.bias
        #print(999)
        
        """
        기존:
        scale = 1 / math.sqrt(self.channels / self.num_heads)
        query_proj = self.reshape_heads_to_batch_dim(query_proj)
        key_proj = self.reshape_heads_to_batch_dim(key_proj)
        value_proj = self.reshape_heads_to_batch_dim(value_proj)
        """
        # 바뀐 버전에서는 scale을 계산할 필요가없음 자동 계산해줌
        # 그리고 reshape_heads_to_batch_dim -> head_to_batch_dim
        query_proj = self.head_to_batch_dim(query_proj)
        key_proj = self.head_to_batch_dim(key_proj)
        value_proj = self.head_to_batch_dim(value_proj)


        attention_scores = torch.baddbmm(
            torch.empty(
                query_proj.shape[0],
                query_proj.shape[1],
                key_proj.shape[1],
                dtype=query_proj.dtype,
                device=query_proj.device,
            ),
            query_proj,
            key_proj.transpose(-1, -2),
            beta=0,
            alpha=self.scale
        )
        attention_probs = torch.softmax(attention_scores.float(), dim=-1).type(attention_scores.dtype)
        hidden_states = torch.bmm(attention_probs, value_proj)

        # reshape hidden_states
        hidden_states = self.batch_to_head_dim(hidden_states)

        # compute next hidden_states
        for layer in self.to_out:
            hidden_states = layer(hidden_states)

        hidden_states = hidden_states.transpose(-1, -2).reshape(batch, channel, height, width)

        # res connect and rescale
        hidden_states = (hidden_states + residual) / self.rescale_output_factor
        return hidden_states

    # Reference: https://github.com/huggingface/diffusers
    def new_forward_vaed(self, z, enconded_fingerprint):
        sample = z
        sample = self.conv_in(sample)
        # middle
        sample = self.mid_block(sample, enconded_fingerprint)
        # up
        for up_block in self.up_blocks:
            sample = up_block(sample, enconded_fingerprint)
        # post-process
        sample = self.conv_norm_out(sample)
        sample = self.conv_act(sample)
        sample = self.conv_out(sample)

        return sample

    @dataclass
    class DecoderOutput(BaseOutput):
        """
        Output of decoding method.
        Args:
            sample (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`):
                Decoded output sample of the model. Output of the last layer of the model.
        """

        sample: torch.FloatTensor

    def new__decode(self, z: torch.FloatTensor, encoded_fingerprint: torch.Tensor, return_dict: bool = True) -> Union[DecoderOutput, torch.FloatTensor]:
        z = self.post_quant_conv(z)
        dec = self.decoder(z, encoded_fingerprint)

        if not return_dict:
            return (dec,)

        return DecoderOutput(sample=dec)

    def new_decode(self, z: torch.FloatTensor, encoded_fingerprint: torch.Tensor, return_dict: bool = True) -> Union[DecoderOutput, torch.FloatTensor]:
        if self.use_slicing and z.shape[0] > 1:
            decoded_slices = [self._decode(z_slice, encoded_fingerprint).sample for z_slice in z.split(1)]
            decoded = torch.cat(decoded_slices)
        else:
            decoded = self._decode(z, encoded_fingerprint).sample

        if not return_dict:
            return (decoded,)

        return DecoderOutput(sample=decoded)

    add_affine_conv(vae.decoder)
    add_affine_attn(vae.decoder)
    #고칠것
    change_forward(vae.decoder, UNetMidBlock2D, new_forward_MB)
    change_forward(vae.decoder, UpDecoderBlock2D, new_forward_UDB)
    change_forward(vae.decoder, ResnetBlock2D, new_forward_RB)
    change_forward(vae.decoder, Attention, new_forward_AB)
    setattr(vae.decoder, 'forward', new_forward_vaed.__get__(vae.decoder, vae.decoder.__class__))
    setattr(vae, '_decode', new__decode.__get__(vae, vae.__class__))
    setattr(vae, 'decode', new_decode.__get__(vae, vae.__class__))

    return vae