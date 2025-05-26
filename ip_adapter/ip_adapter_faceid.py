import json
import os
from typing import List
from venv import logger

import numpy as np
from regex import F
import torch
from torch import nn
from diffusers import StableDiffusionPipeline
from diffusers.pipelines.controlnet import MultiControlNetModel
from PIL import Image
from safetensors import safe_open
from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection
from utils.retrieval_utils import FaissRetriever
from .attention_processor_faceid import LoRAAttnProcessor, FaceLoRAAttentionProcessor
from .utils import is_torch2_available, get_generator

USE_DAFAULT_ATTN = False # should be True for visualization_attnmap

from .resampler import PerceiverAttention, FeedForward


class FacePerceiverResampler(torch.nn.Module):
    def __init__(
        self,
        *,
        dim=768,
        depth=4,
        dim_head=64,
        heads=16,
        embedding_dim=1280,
        output_dim=768,
        ff_mult=4,
    ):
        super().__init__()
        
        self.proj_in = torch.nn.Linear(embedding_dim, dim)
        self.proj_out = torch.nn.Linear(dim, output_dim)
        self.norm_out = torch.nn.LayerNorm(output_dim)
        self.layers = torch.nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                torch.nn.ModuleList(
                    [
                        PerceiverAttention(dim=dim, dim_head=dim_head, heads=heads),
                        FeedForward(dim=dim, mult=ff_mult),
                    ]
                )
            )

    def forward(self, latents, x):
        x = self.proj_in(x)
        for attn, ff in self.layers:
            latents = attn(x, latents) + latents
            latents = ff(latents) + latents
        latents = self.proj_out(latents)
        return self.norm_out(latents)

# 投影faceid特征
class MLPProjModel(torch.nn.Module):
    def __init__(self, cross_attention_dim=768, id_embeddings_dim=512, num_tokens=16):
        super().__init__()
        
        self.cross_attention_dim = cross_attention_dim
        self.num_tokens = num_tokens
        
        self.proj = torch.nn.Sequential(
            torch.nn.Linear(id_embeddings_dim, id_embeddings_dim*2),
            torch.nn.GELU(),
            # torch.nn.Dropout(0.1),  # 添加 dropout
            torch.nn.Linear(id_embeddings_dim*2, cross_attention_dim*num_tokens),
        )
        self.norm = torch.nn.LayerNorm(cross_attention_dim)
        
    def forward(self, id_embeds):
        x = self.proj(id_embeds)
        x = x.reshape(-1, self.num_tokens, self.cross_attention_dim)
        x = self.norm(x)
        return x

# 投影聚合的检索特征
class MLPProjRetrieModel(torch.nn.Module):
    def __init__(self, cross_attention_dim=768, embeddings_dim=768, num_tokens=16):
        super().__init__()
        self.cross_attention_dim = cross_attention_dim
        self.num_tokens = num_tokens
        self.proj = torch.nn.Sequential(
            torch.nn.Linear(embeddings_dim, embeddings_dim*2),
            torch.nn.GELU(),
            torch.nn.Linear(embeddings_dim*2, cross_attention_dim),  # 直接投影到 cross_attention_dim
        )
        self.norm = torch.nn.LayerNorm(cross_attention_dim)

    def forward(self, id_embeds):
        x = self.proj(id_embeds)  # [batch_size, seq_len, cross_attention_dim]
        x = self.norm(x)
        return x
    
# 投影图像clip特征
class ClipProjModel(torch.nn.Module):
    def __init__(self, cross_attention_dim=768, clip_embeddings_dim=768, num_tokens=16):
        super().__init__()
        
        self.cross_attention_dim = cross_attention_dim
        self.clip_embeddings_dim = clip_embeddings_dim
        self.num_tokens = num_tokens
        
        self.proj = torch.nn.Sequential(
            torch.nn.Linear(clip_embeddings_dim, clip_embeddings_dim * 2),  # 768 → 1536
            torch.nn.GELU(),
            torch.nn.Dropout(0.1),  # 添加 dropout
            torch.nn.Linear(clip_embeddings_dim * 2, cross_attention_dim * num_tokens),  # 1536 → cross_attention_dim * 4
        )
        self.norm = torch.nn.LayerNorm(cross_attention_dim)
        
    def forward(self, clip_embeds):
        x = self.proj(clip_embeds)  # (batch_size, cross_attention_dim * num_tokens)
        x = x.reshape(-1, self.num_tokens, self.cross_attention_dim)  # (batch_size, num_tokens, cross_attention_dim)
        x = self.norm(x)
        return x
    
import torch
import torch.nn as nn
import torch.nn.functional as F

class RetrievalFusion(nn.Module):
    def __init__(self, clip_dim=768, id_dim=512, d_model=768, hidden_dim=1024):
        super().__init__()
        self.clip_proj = nn.Sequential(
            nn.Linear(clip_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, d_model)
        )
        self.id_proj = nn.Sequential(
            nn.Linear(id_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, d_model)
        )
        self.norm_clip = nn.LayerNorm(d_model)
        self.norm_id = nn.LayerNorm(d_model)

    def forward(self, retrieval_clip_feat, retrieval_id_feat):
        clip_tokens = self.clip_proj(retrieval_clip_feat)
        clip_tokens = self.norm_clip(clip_tokens)
        id_tokens = self.id_proj(retrieval_id_feat)
        id_tokens = self.norm_id(id_tokens)
        return clip_tokens, id_tokens
    
class RAFAAggregator(nn.Module):
    def __init__(self, d_model=768, nhead=4, dim_feedforward=1024, dropout=0.1, num_tokens=16, token_dropout_rate=0.1, enable_token_shuffle=True):
        super().__init__()
        self.transformer_clip = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                batch_first=True
            ),
            num_layers=2
        )
        self.transformer_id = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                batch_first=True
            ),
            num_layers=2
        )
        self.attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.query_proj = nn.Linear(d_model, d_model)
        self.key_proj = nn.Linear(d_model, d_model)
        self.value_proj = nn.Linear(d_model, d_model)
        self.query_embed = nn.Parameter(torch.zeros(1, num_tokens, d_model))
        nn.init.normal_(self.query_embed, mean=0.0, std=0.02)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.lambda_clip = nn.Parameter(torch.tensor(0.7))
        self.lambda_id = nn.Parameter(torch.tensor(0.3))
        self.d_model = d_model
        self.num_tokens = num_tokens
        self.token_dropout_rate = token_dropout_rate
        self.enable_token_shuffle = enable_token_shuffle

    def forward(self, clip_features, id_features, query):
        batch_size, seq_len_clip, d_model = clip_features.shape
        batch_size_id, seq_len_id, d_model_id = id_features.shape
        assert d_model == self.d_model and d_model_id == self.d_model
        assert batch_size == query.shape[0] == batch_size_id

        if self.training:
            mask_clip = torch.rand(batch_size, seq_len_clip, device=clip_features.device) > self.token_dropout_rate
            mask_clip = mask_clip.unsqueeze(-1)
            clip_features = clip_features * mask_clip

            mask_id = torch.rand(batch_size, seq_len_id, device=id_features.device) > self.token_dropout_rate
            mask_id = mask_id.unsqueeze(-1)
            id_features = id_features * mask_id

            if self.enable_token_shuffle:
                perm_clip = torch.randperm(seq_len_clip, device=clip_features.device)
                clip_features = clip_features[:, perm_clip, :]
                perm_id = torch.randperm(seq_len_id, device=id_features.device)
                id_features = id_features[:, perm_id, :]

        clip_out = self.transformer_clip(clip_features)
        id_out = self.transformer_id(id_features)

        weights = torch.softmax(torch.stack([self.lambda_clip, self.lambda_id]), dim=0)
        clip_weight = weights[0]
        id_weight = weights[1]
        clip_out = clip_out * clip_weight
        id_out = id_out * id_weight

        combined_features = torch.cat([clip_out, id_out], dim=1)

        query = self.query_proj(query.unsqueeze(1))
        query_tokens = self.query_embed + query
        query_tokens = query_tokens.squeeze(1)  # 去除第一维，获得 [batch_size, num_tokens, d_model]

        keys = self.key_proj(combined_features)
        values = self.value_proj(combined_features)
        attn_output, _ = self.attn(query_tokens, keys, values)

        F_agg = self.dropout(attn_output)
        F_agg = self.norm(F_agg)

        return F_agg

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
class IPRetrievalFaceID:
    def __init__(self, sd_pipe: StableDiffusionPipeline, ip_ckpt: str, retriever, device: str, lora_rank: int = 128, num_tokens: int = 16, torch_dtype=torch.float16):
        self.device = device
        self.ip_ckpt = ip_ckpt
        self.lora_rank = lora_rank
        self.num_tokens = num_tokens
        self.torch_dtype = torch_dtype
        self.retriever = retriever

        self.pipe = sd_pipe.to(self.device, dtype=self.torch_dtype)
        
        # 初始化模型，与训练代码保持一致
        cross_attention_dim = self.pipe.unet.config.cross_attention_dim
        
        self.id_proj_high = MLPProjModel(
            cross_attention_dim=cross_attention_dim,
            id_embeddings_dim=512,
            num_tokens=self.num_tokens
        ).to(self.device, dtype=self.torch_dtype)
        
        self.id_proj_low = MLPProjModel(
            cross_attention_dim=cross_attention_dim,
            id_embeddings_dim=512,
            num_tokens=self.num_tokens
        ).to(self.device, dtype=self.torch_dtype)
        
        self.retrieval_proj_model = MLPProjRetrieModel(
            cross_attention_dim=cross_attention_dim,
            embeddings_dim=768,
            num_tokens=self.num_tokens
        ).to(self.device, dtype=self.torch_dtype)
        
        self.transformer = RAFAAggregator(
            d_model=cross_attention_dim,  # 动态设置 d_model
            nhead=4,
            dim_feedforward=1024,
            num_tokens=self.num_tokens
        ).to(self.device, dtype=self.torch_dtype)
        
        self.retrieval_fusion = RetrievalFusion(
            clip_dim=768,
            id_dim=512,
            d_model=cross_attention_dim,  # 动态设置 d_model
            hidden_dim=1024
        ).to(self.device, dtype=self.torch_dtype)

        self.set_ip_adapter()

        if ip_ckpt:
            self.load_ip_adapter()

    def set_ip_adapter(self):
        unet = self.pipe.unet
        attn_procs = {}
        unet_sd = unet.state_dict()
        for name in unet.attn_processors.keys():
            cross_attention_dim = None if name.endswith("attn1.processor") else unet.config.cross_attention_dim
            if name.startswith("mid_block"):
                hidden_size = unet.config.block_out_channels[-1]
            elif name.startswith("up_blocks"):
                block_id = int(name[len("up_blocks.")])
                hidden_size = list(reversed(unet.config.block_out_channels))[block_id]
            elif name.startswith("down_blocks"):
                block_id = int(name[len("down_blocks.")])
                hidden_size = unet.config.block_out_channels[block_id]
            if cross_attention_dim is None:
                attn_procs[name] = LoRAAttnProcessor(
                    hidden_size=hidden_size,
                    cross_attention_dim=cross_attention_dim,
                    rank=self.lora_rank
                ).to(self.device, dtype=self.torch_dtype)
            else:
                layer_name = name.split(".processor")[0]
                weights = {
                    "to_k_ip.weight": unet_sd[layer_name + ".to_k.weight"],
                    "to_v_ip.weight": unet_sd[layer_name + ".to_v.weight"],
                }
                attn_procs[name] = FaceLoRAAttentionProcessor(
                    hidden_size=hidden_size,
                    cross_attention_dim=cross_attention_dim,
                    rank=self.lora_rank,
                    num_tokens=self.num_tokens,
                    num_heads=8,
                    use_residual=True,
                    dropout_p=0.1
                ).to(self.device, dtype=self.torch_dtype)
                attn_procs[name].load_state_dict(weights, strict=False)
        unet.set_attn_processor(attn_procs)
        self.adapter_modules = nn.ModuleList(unet.attn_processors.values())

    def load_ip_adapter(self):
        state_dict = {}
        if os.path.splitext(self.ip_ckpt)[-1] == ".bin":
            state_dict = torch.load(self.ip_ckpt, map_location="cpu")
        else:
            raise ValueError(f"Unsupported checkpoint format: {self.ip_ckpt}. Expected .bin file.")

        logger.info(f"Loaded state_dict keys: {list(state_dict.keys())}")

        # 兼容旧格式（仅包含 ip_adapter 和 image_proj）
        if "ip_adapter" in state_dict and "image_proj" in state_dict:
            self.adapter_modules.load_state_dict(state_dict["ip_adapter"], strict=False)
            self.id_proj_high.load_state_dict(state_dict["image_proj"], strict=True)
            logger.info(f"Loaded legacy ip_adapter and image_proj from {self.ip_ckpt}")
        else:
            # 加载 id_proj_high
            if "id_proj_high" in state_dict and state_dict["id_proj_high"]:
                self.id_proj_high.load_state_dict(state_dict["id_proj_high"], strict=True)
                logger.info(f"Loaded id_proj_high weights from {self.ip_ckpt}")
            else:
                logger.warning(f"'id_proj_high' not found or empty in {self.ip_ckpt}, using random initialization")

            # 加载 id_proj_low
            if "id_proj_low" in state_dict and state_dict["id_proj_low"]:
                self.id_proj_low.load_state_dict(state_dict["id_proj_low"], strict=True)
                logger.info(f"Loaded id_proj_low weights from {self.ip_ckpt}")
            else:
                logger.warning(f"'id_proj_low' not found or empty in {self.ip_ckpt}, using random initialization")

            # 加载 retrieval_proj
            if "retrieval_proj" in state_dict and state_dict["retrieval_proj"]:
                self.retrieval_proj_model.load_state_dict(state_dict["retrieval_proj"], strict=True)
                logger.info(f"Loaded retrieval_proj weights from {self.ip_ckpt}")
            else:
                logger.warning(f"'retrieval_proj' not found or empty in {self.ip_ckpt}, using random initialization")

            # 加载 transformer
            if "transformer" in state_dict and state_dict["transformer"]:
                self.transformer.load_state_dict(state_dict["transformer"], strict=True)
                logger.info(f"Loaded transformer weights from {self.ip_ckpt}")
            else:
                logger.warning(f"'transformer' not found or empty in {self.ip_ckpt}, using random initialization")

            # 加载 retrieval_fusion
            if "retrieval_fusion" in state_dict and state_dict["retrieval_fusion"]:
                self.retrieval_fusion.load_state_dict(state_dict["retrieval_fusion"], strict=True)
                logger.info(f"Loaded retrieval_fusion weights from {self.ip_ckpt}")
            else:
                logger.warning(f"'retrieval_fusion' not found or empty in {self.ip_ckpt}, using random initialization")

            # 加载 ip_adapter
            if "ip_adapter" in state_dict and state_dict["ip_adapter"]:
                self.adapter_modules.load_state_dict(state_dict["ip_adapter"], strict=False)
                logger.info(f"Loaded ip_adapter weights from {self.ip_ckpt}")
            else:
                logger.warning(f"'ip_adapter' not found or empty in {self.ip_ckpt}, using random initialization")

    @torch.inference_mode()
    def get_image_embeds(self, faceid_embeds: torch.Tensor, retrieval_id_features: torch.Tensor, top_k_image_paths: list, feature_db_path: str):
        faceid_embeds = faceid_embeds.to(self.device, dtype=self.torch_dtype)
        retrieval_id_features = retrieval_id_features.to(self.device, dtype=self.torch_dtype)
        
        # 生成输入图的 ID 嵌入（与训练一致，添加归一化）
        low_id_tokens = F.normalize(self.id_proj_low(faceid_embeds), dim=-1)  # [bs, num_tokens, cross_attention_dim]
        uncond_low_id_tokens = F.normalize(self.id_proj_low(torch.zeros_like(faceid_embeds)), dim=-1)
        
        # 加载 CLIP 特征
        retrieval_clip_features_path = os.path.join(feature_db_path, "img_feature", "retrieval_features.npy")
        retrieval_clip_mapping_path = os.path.join(feature_db_path, "img_feature", "retrieval_feature_mapping.json")
        if not os.path.exists(retrieval_clip_features_path) or not os.path.exists(retrieval_clip_mapping_path):
            raise FileNotFoundError(f"CLIP feature files missing: {retrieval_clip_features_path}, {retrieval_clip_mapping_path}")
        retrieval_clip_features = np.load(retrieval_clip_features_path).astype(np.float32)
        with open(retrieval_clip_mapping_path, 'r') as f:
            retrieval_clip_mapping = json.load(f)
        
        # 提取 top-k 个图片的 CLIP 特征
        bs = faceid_embeds.size(0)
        top_k = len(top_k_image_paths[0]) if top_k_image_paths else 0
        clip_features_list = []
        for image_paths in top_k_image_paths:
            clip_features = []
            for img_path in image_paths:
                idx = retrieval_clip_mapping.get(img_path, -1)
                if idx < 0:
                    logger.warning(f"CLIP feature not found for {img_path}, using zero vector")
                    clip_features.append(np.zeros(768, dtype=np.float32))
                else:
                    clip_features.append(retrieval_clip_features[idx])
            clip_features_list.append(np.stack(clip_features))
        retrieval_clip_features_tensor = torch.from_numpy(np.stack(clip_features_list)).to(self.device, dtype=self.torch_dtype)  # [bs, top_k, 768]
        logger.info(f"Retrieved CLIP features: shape={retrieval_clip_features_tensor.shape}, expected=[{bs}, {top_k}, 768]")
        
        # 验证形状
        if retrieval_clip_features_tensor.shape != (bs, top_k, 768):
            raise ValueError(f"Unexpected retrieval_clip_features shape: {retrieval_clip_features_tensor.shape}, expected [{bs}, {top_k}, 768]")
        
        # 使用 RetrievalFusion 融合 CLIP 和 ID 特征
        retrieval_clip_tokens, retrieval_id_tokens = self.retrieval_fusion(retrieval_clip_features_tensor, retrieval_id_features)
        uncond_retrieval_clip_tokens, uncond_retrieval_id_tokens = self.retrieval_fusion(
            torch.zeros_like(retrieval_clip_features_tensor), torch.zeros_like(retrieval_id_features)
        )
        
        # 使用 TransformerAggregator 融合特征
        agg_tokens = self.transformer(retrieval_clip_tokens, retrieval_id_tokens, low_id_tokens)
        retrieval_tokens = F.normalize(self.retrieval_proj_model(agg_tokens), dim=-1)  # [bs, num_tokens, cross_attention_dim]
        uncond_agg_tokens = self.transformer(uncond_retrieval_clip_tokens, uncond_retrieval_id_tokens, uncond_low_id_tokens)
        uncond_retrieval_tokens = F.normalize(self.retrieval_proj_model(uncond_agg_tokens), dim=-1)
        
        logger.debug(f"low_id_tokens shape: {low_id_tokens.shape}, retrieval_tokens shape: {retrieval_tokens.shape}")
        
        return low_id_tokens, uncond_low_id_tokens, retrieval_tokens, uncond_retrieval_tokens

    def set_scale(self, scale: float):
        for attn_processor in self.pipe.unet.attn_processors.values():
            if isinstance(attn_processor, FaceLoRAAttentionProcessor):
                attn_processor.scale = scale

    @torch.inference_mode()
    def generate(
        self,
        faceid_embeds: torch.Tensor,
        retrieval_id_features: torch.Tensor,
        top_k_image_paths: list,
        feature_db_path: str,
        prompt: str = None,
        negative_prompt: str = None,
        scale: float = 1.0,
        num_samples: int = 4,
        seed: int = None,
        guidance_scale: float = 7.5,
        num_inference_steps: int = 30,
        **kwargs,
    ):
        self.set_scale(scale)

        num_prompts = faceid_embeds.size(0)

        if prompt is None:
            prompt = "best quality, high quality"
        if negative_prompt is None:
            negative_prompt = "monochrome, lowres, bad anatomy, worst quality, low quality"

        if not isinstance(prompt, list):
            prompt = [prompt] * num_prompts
        if not isinstance(negative_prompt, list):
            negative_prompt = [negative_prompt] * num_prompts

        # 获取嵌入
        low_id_tokens, uncond_low_id_tokens, retrieval_tokens, uncond_retrieval_tokens = self.get_image_embeds(
            faceid_embeds, retrieval_id_features, top_k_image_paths, feature_db_path
        )

        # 处理嵌入以匹配 num_samples
        bs_embed, seq_len, _ = low_id_tokens.shape
        low_id_tokens = low_id_tokens.repeat(1, num_samples, 1).view(bs_embed * num_samples, seq_len, -1)
        uncond_low_id_tokens = uncond_low_id_tokens.repeat(1, num_samples, 1).view(bs_embed * num_samples, seq_len, -1)
        retrieval_tokens = retrieval_tokens.repeat(1, num_samples, 1).view(bs_embed * num_samples, seq_len, -1)
        uncond_retrieval_tokens = uncond_retrieval_tokens.repeat(1, num_samples, 1).view(bs_embed * num_samples, seq_len, -1)

        # 编码文本提示
        prompt_embeds_, negative_prompt_embeds_ = self.pipe.encode_prompt(
            prompt,
            device=self.device,
            num_images_per_prompt=num_samples,
            do_classifier_free_guidance=True,
            negative_prompt=negative_prompt,
        )

        # 拼接嵌入
        prompt_embeds = torch.cat([prompt_embeds_, retrieval_tokens, low_id_tokens], dim=1)
        negative_prompt_embeds = torch.cat([negative_prompt_embeds_, uncond_retrieval_tokens, uncond_low_id_tokens], dim=1)

        # 设置随机种子
        generator = torch.Generator(device=self.device)
        if seed is not None:
            generator.manual_seed(seed)

        # 生成图片
        images = self.pipe(
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            generator=generator,
            **kwargs,
        ).images

        return images