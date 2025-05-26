# modified from https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py
import torch
from torch import nn
# import torch.nn as nn
import torch.nn.functional as F

from diffusers.models.lora import LoRALinearLayer

class LoRAAttnProcessor(nn.Module):
    
    """
    默认的注意力处理器，仅支持文本特征的注意力计算，基于 LoRA 增强。
    
    Args:
        hidden_size (int): 注意力隐藏维度。
        cross_attention_dim (int, optional): 交叉注意力输入维度，默认与 hidden_size 一致。
        rank (int): LoRA 低秩维度，默认 4。
        lora_scale (float): LoRA 缩放系数，默认 1.0。
        network_alpha (float, optional): LoRA 的网络 alpha 参数。
    """
    
    def __init__(
        self,
        hidden_size=None,
        cross_attention_dim=None,
        rank=4,
        network_alpha=None,
        lora_scale=1.0,
    ):
        super().__init__()
        
        self.hidden_size = hidden_size  # 保存 hidden_size 为实例属性
        self.rank = rank
        self.lora_scale = lora_scale
        
        self.to_q_lora = LoRALinearLayer(hidden_size, hidden_size, rank, network_alpha)
        self.to_k_lora = LoRALinearLayer(cross_attention_dim or hidden_size, hidden_size, rank, network_alpha)
        self.to_v_lora = LoRALinearLayer(cross_attention_dim or hidden_size, hidden_size, rank, network_alpha)
        self.to_out_lora = LoRALinearLayer(hidden_size, hidden_size, rank, network_alpha)

    def __call__(
        self,
        attn,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
        temb=None,
        scale=1.0,  # 添加 scale 参数以匹配 FaceLoRAAttentionProcessor 的签名
        *args,
        **kwargs
    ):
        # 从kwargs中弹出不需要的参数以避免警告
        _ = kwargs.pop("ip_hidden_states", None)
        _ = kwargs.pop("retrieval_hidden_states", None)
        residual = hidden_states

        # 预处理：空间归一化和维度调整
        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)
        input_ndim = hidden_states.ndim
        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)
        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )
        attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        # 文本支路
        query = attn.to_q(hidden_states) + self.lora_scale * self.to_q_lora(hidden_states)
        inner_dim = self.hidden_size
        head_dim = inner_dim // attn.heads
        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        # 处理 encoder_hidden_states
        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states  # 自注意力情况
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states) + self.lora_scale * self.to_k_lora(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states) + self.lora_scale * self.to_v_lora(encoder_hidden_states)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        # 计算注意力
        attn_output = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )
        attn_output = attn_output.transpose(1, 2).reshape(batch_size, -1, inner_dim).to(query.dtype)

        # 输出投影
        attn_output = attn.to_out[0](attn_output) + self.lora_scale * self.to_out_lora(attn_output)
        attn_output = attn.to_out[1](attn_output)  # dropout

        # 后处理：恢复输入维度
        if input_ndim == 4:
            attn_output = attn_output.transpose(-1, -2).reshape(batch_size, channel, height, width)
        if attn.residual_connection:
            attn_output = attn_output + residual
        attn_output = attn_output / attn.rescale_output_factor

        return attn_output

class FaceLoRAAttentionProcessor(nn.Module):
    def __init__(
        self,
        hidden_size=None,
        cross_attention_dim=None,
        rank=4,
        network_alpha=None,
        lora_scale=1.0,
        num_tokens=16,
        num_heads=8,
        use_residual=True,
        dropout_p=0.05
    ):
        """
        初始化 FaceLoRAAttentionProcessor，支持三路条件融合（Text、Retrieval、ID）。

        参数：
            hidden_size (int): 隐藏状态维度。
            cross_attention_dim (int): 跨注意力维度，默认为 hidden_size。
            rank (int): LoRA 层的秩。
            network_alpha (float): LoRA 网络的 alpha 参数。
            lora_scale (float): LoRA 缩放系数。
            num_tokens (int): 每个分支的 token 数量。
            num_heads (int): 注意力头数。
            use_residual (bool): 是否使用残差连接。
            dropout_p (float): Dropout 概率。
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.cross_attention_dim = cross_attention_dim or hidden_size
        self.rank = rank
        self.lora_scale = lora_scale
        self.num_tokens = num_tokens
        self.num_heads = num_heads
        self.use_residual = use_residual
        self.head_dim = hidden_size // num_heads
        self.seq_len_text = 77  # CLIP 默认文本序列长度

        # 计算缩放因子
        self.attn_scale = self.head_dim ** -0.5

        # 主干投影
        self.to_q = nn.Linear(hidden_size, hidden_size, bias=False)
        
        # 文本分支投影（无 LoRA）
        self.to_k_text = nn.Linear(self.cross_attention_dim, hidden_size, bias=False)
        self.to_v_text = nn.Linear(self.cross_attention_dim, hidden_size, bias=False)
        
        # 检索分支投影
        self.to_k_retrieval = nn.Linear(self.cross_attention_dim, hidden_size, bias=False)
        self.to_v_retrieval = nn.Linear(self.cross_attention_dim, hidden_size, bias=False)
        
        # ID分支投影
        self.to_k_ip = nn.Linear(self.cross_attention_dim, hidden_size, bias=False)
        self.to_v_ip = nn.Linear(self.cross_attention_dim, hidden_size, bias=False)

        # LoRA 层：为 Retrieval 和 ID 分支添加可训练的低秩更新
        self.to_k_retrieval_lora = LoRALinearLayer(self.cross_attention_dim, hidden_size, rank, network_alpha)
        self.to_v_retrieval_lora = LoRALinearLayer(self.cross_attention_dim, hidden_size, rank, network_alpha)
        self.to_k_ip_lora = LoRALinearLayer(self.cross_attention_dim, hidden_size, rank, network_alpha)
        self.to_v_ip_lora = LoRALinearLayer(self.cross_attention_dim, hidden_size, rank, network_alpha)

        # 输出投影
        self.to_out = nn.Linear(hidden_size, hidden_size, bias=False)
        self.to_out_lora = LoRALinearLayer(hidden_size, hidden_size, rank, network_alpha)

        # 可学习融合权重
        self.lambda_id = nn.Parameter(torch.tensor(0.5))       # Retrieval 和 ID 融合权重
        self.lambda_final = nn.Parameter(torch.tensor(0.5))    # 最终融合权重
        self.lambda_residual = nn.Parameter(torch.tensor(0.5)) # 残差融合权重

        # Dropout 层
        self.dropout = nn.Dropout(dropout_p)

    def __call__(
        self,
        attn,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
        temb=None,
        retrieval_hidden_states=None,
        id_hidden_states=None,
        *args,
        **kwargs
    ):
        """
        前向传播：三路条件融合（Text、Retrieval、ID）。

        参数：
            attn: 原始注意力模块。
            hidden_states (torch.Tensor): 输入隐藏状态 [batch_size, seq_len, hidden_size]。
            encoder_hidden_states (torch.Tensor): 编码器隐藏状态 [batch_size, seq_len_text + 2*num_tokens, cross_attention_dim]。
            attention_mask (torch.Tensor): 注意力掩码。
            temb (torch.Tensor): 时间嵌入。
            retrieval_hidden_states (torch.Tensor): 检索特征 [batch_size, num_tokens, cross_attention_dim]。
            id_hidden_states (torch.Tensor): ID特征 [batch_size, num_tokens, cross_attention_dim]。
        """
        residual = hidden_states

        # 预处理：空间归一化和维度调整
        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)
        input_ndim = hidden_states.ndim
        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)
        batch_size, seq_len, _ = hidden_states.shape
        attention_mask = attn.prepare_attention_mask(attention_mask, seq_len, batch_size)
        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        # 查询投影
        query = self.to_q(hidden_states)
        query = query.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # 从 encoder_hidden_states 分割特征
        text_hidden_states = None
        if encoder_hidden_states is not None:
            text_hidden_states = encoder_hidden_states[:, :self.seq_len_text, :]
            retrieval_hidden_states = encoder_hidden_states[:, self.seq_len_text:self.seq_len_text + self.num_tokens, :]
            id_hidden_states = encoder_hidden_states[:, self.seq_len_text + self.num_tokens:self.seq_len_text + 2 * self.num_tokens, :]

        # 处理缺失输入
        if text_hidden_states is None:
            text_hidden_states = torch.zeros(batch_size, self.seq_len_text, self.cross_attention_dim, device=hidden_states.device)
        if retrieval_hidden_states is None:
            retrieval_hidden_states = torch.zeros(batch_size, self.num_tokens, self.cross_attention_dim, device=hidden_states.device)
        if id_hidden_states is None:
            id_hidden_states = torch.zeros(batch_size, self.num_tokens, self.cross_attention_dim, device=hidden_states.device)

        # 各分支键值投影
        if attn.norm_cross:
            text_hidden_states = attn.norm_encoder_hidden_states(text_hidden_states)
        key_text = self.to_k_text(text_hidden_states)
        value_text = self.to_v_text(text_hidden_states)

        key_retrieval = self.to_k_retrieval(retrieval_hidden_states) + self.lora_scale * self.to_k_retrieval_lora(retrieval_hidden_states)
        value_retrieval = self.to_v_retrieval(retrieval_hidden_states) + self.lora_scale * self.to_v_retrieval_lora(retrieval_hidden_states)

        key_id = self.to_k_ip(id_hidden_states) + self.lora_scale * self.to_k_ip_lora(id_hidden_states)
        value_id = self.to_v_ip(id_hidden_states) + self.lora_scale * self.to_v_ip_lora(id_hidden_states)

        # 第一阶段：融合 Retrieval 和 ID 特征
        lambda_id = torch.clamp(self.lambda_id, 0.0, 1.0)
        key_visual = lambda_id * key_retrieval + (1 - lambda_id) * key_id
        value_visual = lambda_id * value_retrieval + (1 - lambda_id) * value_id

        # 第二阶段：融合视觉特征和文本
        lambda_final = torch.clamp(self.lambda_final, 0.0, 1.0)
        key = torch.cat([(1 - lambda_final) * key_text, lambda_final * key_visual], dim=1)
        value = torch.cat([(1 - lambda_final) * value_text, lambda_final * value_visual], dim=1)

        # 注意力计算
        key = key.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(query, key.transpose(-1, -2)) * self.attn_scale
        if attention_mask is not None:
            scores += attention_mask

        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        out = torch.matmul(attn_weights, value)

        out = out.transpose(1, 2).reshape(batch_size, seq_len, self.hidden_size)
        out = self.to_out(out) + self.lora_scale * self.to_out_lora(out)
        out = self.dropout(out)

        # 残差连接
        if self.use_residual:
            lambda_residual = torch.clamp(self.lambda_residual, 0.0, 1.0)
            out = lambda_residual * residual + (1 - lambda_residual) * out
        else:
            if attn.residual_connection:
                out = out + residual

        # 后处理
        if input_ndim == 4:
            out = out.transpose(-1, -2).reshape(batch_size, self.hidden_size, height, width)
        out = out / attn.rescale_output_factor

        return out