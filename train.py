import os
import random
import argparse
from pathlib import Path
import json
import itertools
import logging
import lpips
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
from transformers import CLIPTextModel, CLIPTokenizer
from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration
from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
from ip_adapter.ip_adapter_faceid import MLPProjModel, RAFAAggregator, RetrievalFusion, MLPProjRetrieModel
from ip_adapter.attention_processor_faceid import FaceLoRAAttentionProcessor, LoRAAttnProcessor
from utils.retrieval_utils import FaissRetriever
from accelerate.utils import DistributedDataParallelKwargs
import xformers

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

class MyDataset(torch.utils.data.Dataset):
    def __init__(self, json_file, tokenizer, size=512, t_drop_rate=0.05, i_drop_rate=0.05, ti_drop_rate=0.01, image_root_path="", feature_db_path="", split="train"):
        super().__init__()
        self.tokenizer = tokenizer
        self.size = size
        self.t_drop_rate = t_drop_rate
        self.i_drop_rate = i_drop_rate
        self.ti_drop_rate = ti_drop_rate
        self.image_root_path = image_root_path
        self.feature_db_path = feature_db_path
        self.split = split

        if not os.path.exists(json_file):
            raise FileNotFoundError(f"Data JSON file not found: {json_file}")
        with open(json_file, 'r') as f:
            data = json.load(f)
        self.data = [item for item in data if item.get("split", "train") == split]
        if not self.data:
            raise ValueError(f"No data found for {split} split in {json_file}")
        logger.info(f"Loaded {len(self.data)} items for {split} split from {json_file}")

        self.transform = transforms.Compose([
            transforms.Resize(self.size, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
            transforms.CenterCrop(self.size),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
        
        self.tokenizer.pad_token = self.tokenizer.pad_token or self.tokenizer.eos_token

        self.feature_dirs = {
            "train_data": os.path.join(self.feature_db_path, "train_data"),
            "low": os.path.join(self.feature_db_path, "low"),
            "retrieval": os.path.join(self.feature_db_path, "retrieval"),
            "clip_high": os.path.join(self.feature_db_path, "img_feature"),
        }

        self.features = {}
        self.key_mappings = {}
        for db_name, db_path in self.feature_dirs.items():
            if db_name in ["train_data", "low", "retrieval"]:
                features_path = os.path.join(db_path, "all_features.npy")
                mapping_path = os.path.join(db_path, "face_key_mapping.json")
            elif db_name == "clip_high":
                features_path = os.path.join(db_path, "high_features.npy")  # 修正为 high_features.npy
                mapping_path = os.path.join(db_path, "high_feature_mapping.json")
            
            if not os.path.exists(features_path) or not os.path.exists(mapping_path):
                raise FileNotFoundError(f"Feature file or mapping missing: {features_path}, {mapping_path}")
            
            self.features[db_name] = np.load(features_path).astype(np.float32)
            with open(mapping_path, 'r') as f:
                self.key_mappings[db_name] = json.load(f)
            logger.info(f"Loaded {db_name}: {self.features[db_name].shape[0]} features, shape: {self.features[db_name].shape}")

        filtered_data = []
        skipped_count = 0
        for item in self.data:
            low_image_file = item.get("low_quality_file")
            high_image_file = item.get("image_file")
            if not low_image_file or not high_image_file:
                skipped_count += 1
                logger.debug(f"Skipping item with missing low_quality_file or image_file")
                continue
            low_key = low_image_file
            high_key = high_image_file
            low_idx = self.key_mappings["low"].get(low_key, -1)
            high_idx = self.key_mappings["train_data"].get(high_key, -1)
            high_clip_idx = self.key_mappings["clip_high"].get(high_key, -1)
            
            if low_idx >= 0 and high_idx >= 0 and high_clip_idx >= 0:
                filtered_data.append(item)
            else:
                skipped_count += 1
                logger.debug(f"Skipping item: low_idx={low_idx}, high_idx={high_idx}, high_clip_idx={high_clip_idx}")
        
        self.data = filtered_data
        if not self.data:
            raise ValueError(f"Dataset is empty after filtering for {split} split, skipped {skipped_count} items")
        logger.info(f"After filtering, {len(self.data)} items remain for {split} split, skipped {skipped_count} items")

    def __getitem__(self, idx):
        item = self.data[idx]
        text = item.get("text", "")
        low_image_file = item["low_quality_file"]
        high_image_file = item["image_file"]

        high_key = high_image_file
        high_idx = self.key_mappings["train_data"].get(high_key, -1)
        if high_idx < 0:
            raise ValueError(f"High-quality image {high_image_file} not found in train_data mapping")
        high_id_features = torch.from_numpy(self.features["train_data"][high_idx])  # [512]

        low_key = low_image_file
        low_idx = self.key_mappings["low"].get(low_key, -1)
        if low_idx < 0:
            raise ValueError(f"Low-quality image {low_image_file} not found in low mapping")
        low_id_features = torch.from_numpy(self.features["low"][low_idx])  # [512]

        high_clip_idx = self.key_mappings["clip_high"].get(high_key, -1)
        if high_clip_idx < 0:
            raise ValueError(f"High-quality CLIP feature {high_image_file} not found in clip_high mapping")
        high_clip_features = torch.from_numpy(self.features["clip_high"][high_clip_idx])  # [768]

        low_image_path = os.path.join(self.image_root_path, low_image_file)
        high_image_path = os.path.join(self.image_root_path, high_image_file)
        if not os.path.exists(low_image_path) or not os.path.exists(high_image_path):
            raise FileNotFoundError(f"Image file missing: {low_image_path} or {high_image_path}")
        low_image = Image.open(low_image_path).convert("RGB")
        high_image = Image.open(high_image_path).convert("RGB")
        low_image = self.transform(low_image)
        high_image = self.transform(high_image)

        # 优化 dropout 逻辑，避免重复应用
        if random.random() < self.ti_drop_rate:
            text = ""
            low_id_features = torch.zeros_like(low_id_features)
        elif random.random() < self.t_drop_rate:
            text = ""
        elif random.random() < self.i_drop_rate:
            low_id_features = torch.zeros_like(low_id_features)

        text_input_ids = self.tokenizer(
            text, max_length=self.tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt"
        ).input_ids[0]

        return {
            "low_image": low_image,
            "high_image": high_image,
            "text_input_ids": text_input_ids,
            "low_id_features": low_id_features,
            "high_id_features": high_id_features,
            "high_clip_features": high_clip_features,
        }
    
    def __len__(self):
        return len(self.data)

def collate_fn(data):
    return {
        "low_images": torch.stack([example["low_image"] for example in data]),
        "high_images": torch.stack([example["high_image"] for example in data]),
        "text_input_ids": torch.stack([example["text_input_ids"].squeeze(0) for example in data], dim=0),
        "low_id_features": torch.stack([example["low_id_features"] for example in data]),
        "high_id_features": torch.stack([example["high_id_features"] for example in data]),
        "high_clip_features": torch.stack([example["high_clip_features"] for example in data]),
    }

class IPRetrieval(nn.Module):
    def __init__(self, unet, adapter_modules, retrieval_proj_model, id_proj_high, id_proj_low, transformer, ckpt_path=None):
        super().__init__()
        self.unet = unet
        self.adapter_modules = adapter_modules
        self.retrieval_proj_model = retrieval_proj_model
        self.id_proj_high = id_proj_high
        self.id_proj_low = id_proj_low
        self.transformer = transformer
        self.retrieval_fusion = RetrievalFusion(
            clip_dim=768, id_dim=512, d_model=unet.config.cross_attention_dim, hidden_dim=1024
        )
        self.p_drop_ret = 0.05
        self.id_proj_high.requires_grad_(False)
        if ckpt_path:
            self.load_from_checkpoint(ckpt_path)

    def forward(self, noisy_latents, timesteps, encoder_hidden_states, low_id_features, high_id_features, retrieval_clip_features, retrieval_id_features):
        high_id_tokens = F.normalize(self.id_proj_high(high_id_features), dim=-1)  # [B, T, D]
        low_id_tokens = F.normalize(self.id_proj_low(low_id_features), dim=-1)  # [B, T, D]
        
        retrieval_clip_tokens, retrieval_id_tokens = self.retrieval_fusion(retrieval_clip_features, retrieval_id_features)  # [B, top_k, D]
        
        agg_tokens = self.transformer(retrieval_clip_tokens, retrieval_id_tokens, low_id_tokens)  # [B, T, D]
        if self.training and torch.rand(1) < self.p_drop_ret:
            batch_size, num_tokens, d_model = agg_tokens.shape
            mask = torch.ones(batch_size, num_tokens, device=agg_tokens.device)
            mask_indices = torch.rand(batch_size, num_tokens, device=agg_tokens.device)
            mask[mask_indices < 0.5] = 0
            mask = mask.unsqueeze(-1)
            agg_tokens = agg_tokens * mask
        retrieval_tokens = F.normalize(self.retrieval_proj_model(agg_tokens), dim=-1)  # [B, T, D]
        
        combined_hidden_states = torch.cat([encoder_hidden_states, retrieval_tokens, low_id_tokens], dim=1)
        noise_pred = self.unet(sample=noisy_latents, timestep=timesteps, encoder_hidden_states=combined_hidden_states).sample
       
        return noise_pred, retrieval_tokens, high_id_tokens, low_id_tokens

    def load_from_checkpoint(self, ckpt_path):
        state_dict = torch.load(ckpt_path, map_location="cpu")
        self.adapter_modules.load_state_dict(state_dict["ip_adapter"], strict=False)
        self.id_proj_high.load_state_dict(state_dict["image_proj"], strict=True)
        logger.info(f"Loaded weights from {ckpt_path}")

def parse_args():
    parser = argparse.ArgumentParser(description="Training script for IP-Retrieval.")
    parser.add_argument("--pretrained_model_name_or_path", type=str, required=True)
    parser.add_argument("--pretrained_ip_adapter_path", type=str, default=None)
    parser.add_argument("--data_json_file", type=str, required=True)
    parser.add_argument("--data_root_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="sd-ip_adapter")
    parser.add_argument("--logging_dir", type=str, default="logs")
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-2)
    parser.add_argument("--num_train_epochs", type=int, default=100)
    parser.add_argument("--train_batch_size", type=int, default=4)
    parser.add_argument("--dataloader_num_workers", type=int, default=1)
    parser.add_argument("--save_steps", type=int, default=2000)
    parser.add_argument("--validate_steps", type=int, default=500)
    parser.add_argument("--mixed_precision", type=str, default="fp16", choices=["no", "fp16", "bf16"])
    parser.add_argument("--report_to", type=str, default="tensorboard")
    parser.add_argument("--local_rank", type=int, default=-1)
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lambda_retrieval", type=float, default=0.1)
    parser.add_argument("--lambda_mse", type=float, default=1.0)
    parser.add_argument("--lambda_id", type=float, default=0.1)
    parser.add_argument("--lambda_lpips", type=float, default=0.2)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--feature_db_path", type=str, default="")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None)
    parser.add_argument("--tau", type=float, default=0.07)
    
    args = parser.parse_args()
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank
    return args

def cosine_similarity_loss(x, y, dim=-1, eps=1e-8):
    x_norm = x / (x.norm(dim=dim, keepdim=True) + eps)
    y_norm = y / (y.norm(dim=dim, keepdim=True) + eps)
    cos_sim = (x_norm * y_norm).sum(dim=dim)
    return 1 - cos_sim.mean()

def train(args, accelerator, train_dataloader, val_dataloader, ip_retrieval, noise_scheduler, vae, text_encoder, retriever):
    
    def compute_id_align_loss(low_id_tokens, high_id_tokens, temperature=args.tau):
        local_B = low_id_tokens.size(0)
        anchor = F.normalize(low_id_tokens.mean(dim=1), dim=-1)  # [B, D]
        positive = F.normalize(high_id_tokens.mean(dim=1), dim=-1)  # [B, D]
        # 在 DDP 中收集全局正样本
        with torch.no_grad():
            gathered_positive = accelerator.gather(positive)  # [global_B, D]
        gathered_positive = gathered_positive.to(anchor.device)
        logits = torch.matmul(anchor, gathered_positive.T) / temperature  # [local_B, global_B]
        labels = torch.arange(local_B, device=logits.device) + (accelerator.process_index * local_B)
        id_align_loss = F.cross_entropy(logits, labels)
        logger.debug(f"id_align_loss: {id_align_loss.item()}, logits shape: {logits.shape}, labels: {labels.tolist()}")
        return id_align_loss

    def compute_retrieval_align_loss(retrieval_tokens, high_clip_features, high_id_features, temperature=args.tau):
        model = accelerator.unwrap_model(ip_retrieval)
        local_B = retrieval_tokens.size(0)
        D = model.retrieval_proj_model.cross_attention_dim  # 768
        T = model.retrieval_proj_model.num_tokens  # 16
        # 处理 high_clip_features 和 high_id_features
        high_clip_feats = high_clip_features.unsqueeze(1)  # [B, 1, 768]
        high_id_feats = high_id_features.unsqueeze(1)  # [B, 1, 512]
        high_clip_tokens, high_id_tokens = model.retrieval_fusion(high_clip_feats, high_id_feats)  # [B, 1, D]
        high_id_tokens_input = F.normalize(model.id_proj_high(high_id_features), dim=-1)  # [B, T, D]
        agg_clip_high = model.transformer(high_clip_tokens, high_id_tokens, high_id_tokens_input)  # [B, T, D]
        high_tokens = F.normalize(model.retrieval_proj_model(agg_clip_high), dim=-1).mean(dim=1)  # [B, D]
        anchor = F.normalize(retrieval_tokens.mean(dim=1), dim=-1)  # [B, D]
        positive = F.normalize(high_tokens, dim=-1)  # [B, D]
        with torch.no_grad():
            gathered_positive = accelerator.gather(positive)  # [global_B, D]
        gathered_positive = gathered_positive.to(anchor.device)
        logits = torch.matmul(anchor, gathered_positive.T) / temperature  # [local_B, global_B]
        labels = torch.arange(local_B, device=logits.device) + (accelerator.process_index * local_B)
        retrieval_align_loss = F.cross_entropy(logits, labels)
        logger.debug(f"retrieval_align_loss: {retrieval_align_loss.item()}, logits shape: {logits.shape}, labels: {labels.tolist()}")
        return retrieval_align_loss

    retrieval_clip_features_path = os.path.join(args.feature_db_path, "img_feature", "retrieval_features.npy")
    retrieval_clip_mapping_path = os.path.join(args.feature_db_path, "img_feature", "retrieval_feature_mapping.json")
    if not os.path.exists(retrieval_clip_features_path) or not os.path.exists(retrieval_clip_mapping_path):
        raise FileNotFoundError(f"CLIP feature files missing: {retrieval_clip_features_path}, {retrieval_clip_mapping_path}")
    retrieval_clip_features = np.load(retrieval_clip_features_path).astype(np.float32)
    with open(retrieval_clip_mapping_path, 'r') as f:
        retrieval_clip_mapping = json.load(f)
    logger.info(f"Loaded retrieval CLIP features: {retrieval_clip_features.shape} from {retrieval_clip_features_path}")

    params_to_opt = itertools.chain(
        ip_retrieval.id_proj_low.parameters(),
        ip_retrieval.retrieval_proj_model.parameters(),
        ip_retrieval.adapter_modules.parameters(),
        ip_retrieval.transformer.parameters(),
        ip_retrieval.retrieval_fusion.parameters()
    )
    optimizer = torch.optim.AdamW(params_to_opt, lr=args.learning_rate, weight_decay=args.weight_decay)

    ip_retrieval, optimizer, train_dataloader, val_dataloader = accelerator.prepare(
        ip_retrieval, optimizer, train_dataloader, val_dataloader
    )
    
    lpips_model = lpips.LPIPS(net='vgg').to(accelerator.device)

    total_steps = args.num_train_epochs * len(train_dataloader)
    global_step = 0
    
    if accelerator.is_main_process:
        accelerator.init_trackers("ip_retrieval_training")

    if args.resume_from_checkpoint:
        accelerator.print(f"Resuming from checkpoint: {args.resume_from_checkpoint}")
        accelerator.load_state(args.resume_from_checkpoint)
        checkpoint_step = int(args.resume_from_checkpoint.split("checkpoint-")[-1])
        global_step = checkpoint_step
        start_epoch = global_step // len(train_dataloader)
        step_offset = global_step % len(train_dataloader)
        accelerator.print(f"Resuming training from step {global_step} (epoch {start_epoch}, step offset {step_offset})")
    else:
        start_epoch = 0
        step_offset = 0

    for epoch in range(start_epoch, args.num_train_epochs):
        ip_retrieval.train()
        for step, batch in enumerate(train_dataloader):
            if epoch == start_epoch and step < step_offset:
                global_step += 1
                continue

            with accelerator.accumulate(ip_retrieval):
                with torch.no_grad():
                    latents = vae.encode(batch["low_images"].to(accelerator.device)).latent_dist.sample() * vae.config.scaling_factor
                
                noise = torch.randn_like(latents)
                bsz = latents.shape[0]
                timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device).long()
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                with torch.no_grad():
                    encoder_hidden_states = text_encoder(batch["text_input_ids"].to(accelerator.device))[0]

                low_id_features = batch["low_id_features"].to(accelerator.device)
                high_id_features = batch["high_id_features"].to(accelerator.device)
                high_clip_features = batch["high_clip_features"].to(accelerator.device)
               
                _, _, retrieval_id_features, top_k_image_paths = retriever.retrieve_top_k(low_id_features)
                retrieval_id_features = retrieval_id_features.to(accelerator.device)
                
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
                retrieval_clip_features_tensor = torch.from_numpy(np.stack(clip_features_list)).to(accelerator.device)
                
                noise_pred, retrieval_tokens, high_id_tokens, low_id_tokens = ip_retrieval(
                    noisy_latents=noisy_latents,
                    timesteps=timesteps,
                    encoder_hidden_states=encoder_hidden_states,
                    low_id_features=low_id_features,
                    high_id_features=high_id_features,
                    retrieval_clip_features=retrieval_clip_features_tensor,
                    retrieval_id_features=retrieval_id_features
                )

                noise_loss = F.mse_loss(noise_pred, noise)
                id_align_loss = compute_id_align_loss(low_id_tokens, high_id_tokens)
                retrieval_align_loss = compute_retrieval_align_loss(retrieval_tokens, high_clip_features, high_id_features)
                
                generated_latents = []
                for i in range(noise_pred.shape[0]):
                    step_output = noise_scheduler.step(
                        model_output=noise_pred[i].unsqueeze(0),
                        timestep=timesteps[i].unsqueeze(0),
                        sample=noisy_latents[i].unsqueeze(0)
                    )
                    generated_latents.append(step_output.pred_original_sample)  # 统一使用 pred_original_sample
                generated_latents = torch.cat(generated_latents, dim=0)
                
                gen_images = vae.decode(generated_latents / vae.config.scaling_factor, return_dict=False)[0]
                gen_images = (gen_images.clamp(-1, 1) + 1) / 2
                real_images = (batch["high_images"].to(accelerator.device).clamp(-1, 1) + 1) / 2
                lpips_loss = lpips_model(gen_images, real_images).mean()

                loss = (args.lambda_mse * noise_loss +
                        args.lambda_id * id_align_loss +
                        args.lambda_retrieval * retrieval_align_loss +
                        args.lambda_lpips * lpips_loss)

                accelerator.backward(loss)
                optimizer.step()
                optimizer.zero_grad()

                if accelerator.is_main_process and global_step % 100 == 0:
                    progress = (global_step / total_steps) * 100
                    logger.info(f"Epoch {epoch}, Progress: {progress:.2f}% (Step {global_step}/{total_steps}), "
                               f"noise_loss={noise_loss.item():.4f}, id_align_loss={id_align_loss.item():.4f}, "
                               f"retrieval_align_loss={retrieval_align_loss.item():.4f}, lpips_loss={lpips_loss.item():.4f}, "
                               f"total={loss.item():.4f}")
                    accelerator.log({
                        "train/noise_loss": noise_loss.item(),
                        "train/id_align_loss": id_align_loss.item(),
                        "train/retrieval_align_loss": retrieval_align_loss.item(),
                        "train/lpips_loss": lpips_loss.item(),
                        "train/total_loss": loss.item(),
                        "epoch": epoch
                    }, step=global_step)

            global_step += 1
            if global_step % args.save_steps == 0:
                save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                accelerator.save_state(save_path, safe_serialization=False)

            if global_step % args.validate_steps == 0:
                val_noise_loss, val_id_align_loss, val_retrieval_align_loss, val_lpips_loss, val_total_loss = validate(
                    accelerator, val_dataloader, ip_retrieval, noise_scheduler, vae, 
                    text_encoder, retriever, args, lpips_model, retrieval_clip_features, retrieval_clip_mapping
                )
                if accelerator.is_main_process:
                    logger.info(f"Validation at step {global_step}: "
                               f"noise_loss={val_noise_loss:.4f}, id_align_loss={val_id_align_loss:.4f}, "
                               f"retrieval_align_loss={val_retrieval_align_loss:.4f}, lpips_loss={val_lpips_loss:.4f}, "
                               f"total={val_total_loss:.4f}")
                    accelerator.log({
                        "validation/noise_loss": val_noise_loss,
                        "validation/id_align_loss": val_id_align_loss,
                        "validation/retrieval_align_loss": val_retrieval_align_loss,
                        "validation/lpips_loss": val_lpips_loss,
                        "validation/total_loss": val_total_loss
                    }, step=global_step)

    if accelerator.is_main_process:
        save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
        accelerator.save_state(save_path, safe_serialization=False)
        state_dict = {
            "id_proj_high": accelerator.unwrap_model(ip_retrieval).id_proj_high.state_dict(),
            "id_proj_low": accelerator.unwrap_model(ip_retrieval).id_proj_low.state_dict(),
            "retrieval_proj": accelerator.unwrap_model(ip_retrieval).retrieval_proj_model.state_dict(),
            "transformer": accelerator.unwrap_model(ip_retrieval).transformer.state_dict(),
            "ip_adapter": accelerator.unwrap_model(ip_retrieval).adapter_modules.state_dict(),
            "retrieval_fusion": accelerator.unwrap_model(ip_retrieval).retrieval_fusion.state_dict()
        }
        torch.save(state_dict, os.path.join(save_path, "pytorch_model.bin"))
        logger.info(f"Training completed, saved final checkpoint at {save_path}")
        accelerator.end_training()

def validate(accelerator, val_dataloader, ip_retrieval, noise_scheduler, vae, text_encoder, retriever, args, lpips_model, retrieval_clip_features, retrieval_clip_mapping):
    def compute_id_align_loss(low_id_tokens, high_id_tokens, temperature=args.tau):
        local_B = low_id_tokens.size(0)
        anchor = F.normalize(low_id_tokens.mean(dim=1), dim=-1)  # [B, D]
        positive = F.normalize(high_id_tokens.mean(dim=1), dim=-1)  # [B, D]

        with torch.no_grad():
            gathered_positive = accelerator.gather(positive)  # [global_B, D]
        gathered_positive = gathered_positive.to(anchor.device)

        logits = torch.matmul(anchor, gathered_positive.T) / temperature  # [local_B, global_B]
        labels = torch.arange(local_B, device=logits.device) + (accelerator.process_index * local_B)
        id_align_loss = F.cross_entropy(logits, labels)
        logger.debug(f"id_align_loss: {id_align_loss.item()}, logits shape: {logits.shape}, labels: {labels.tolist()}")

        return id_align_loss

    def compute_retrieval_align_loss(retrieval_tokens, high_clip_features, high_id_features, temperature=args.tau):
        model = accelerator.unwrap_model(ip_retrieval)
        local_B = retrieval_tokens.size(0)
        D = model.retrieval_proj_model.cross_attention_dim  # 768
        T = model.retrieval_proj_model.num_tokens  # 16

        high_clip_feats = high_clip_features.unsqueeze(1)  # [B, 1, 768]
        high_id_feats = high_id_features.unsqueeze(1)  # [B, 1, 512]
        high_clip_tokens, high_id_tokens = model.retrieval_fusion(high_clip_feats, high_id_feats)  # [B, 1, D]
        high_id_tokens_input = F.normalize(model.id_proj_high(high_id_features), dim=-1)  # [B, T, D]
        agg_clip_high = model.transformer(high_clip_tokens, high_id_tokens, high_id_tokens_input)  # [B, T, D]
        high_tokens = F.normalize(model.retrieval_proj_model(agg_clip_high), dim=-1).mean(dim=1)  # [B, D]

        anchor = F.normalize(retrieval_tokens.mean(dim=1), dim=-1)  # [B, D]
        positive = F.normalize(high_tokens, dim=-1)  # [B, D]

        with torch.no_grad():
            gathered_positive = accelerator.gather(positive)  # [global_B, D]
        gathered_positive = gathered_positive.to(anchor.device)

        logits = torch.matmul(anchor, gathered_positive.T) / temperature  # [local_B, global_B]
        labels = torch.arange(local_B, device=logits.device) + (accelerator.process_index * local_B)
        retrieval_align_loss = F.cross_entropy(logits, labels)
        logger.debug(f"retrieval_align_loss: {retrieval_align_loss.item()}, logits shape: {logits.shape}, labels: {labels.tolist()}")

        return retrieval_align_loss

    ip_retrieval.eval()
    total_noise_loss = 0.0
    total_id_align_loss = 0.0
    total_retrieval_align_loss = 0.0
    total_lpips_loss = 0.0
    total_loss = 0.0
    num_batches = 0

    with torch.no_grad():
        for batch in val_dataloader:
            latents = vae.encode(batch["low_images"].to(accelerator.device)).latent_dist.sample() * vae.config.scaling_factor
            
            noise = torch.randn_like(latents)
            bsz = latents.shape[0]
            timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device).long()
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

            encoder_hidden_states = text_encoder(batch["text_input_ids"].to(accelerator.device))[0]

            low_id_features = batch["low_id_features"].to(accelerator.device)
            high_id_features = batch["high_id_features"].to(accelerator.device)
            high_clip_features = batch["high_clip_features"].to(accelerator.device)

            _, _, retrieval_id_features, top_k_image_paths = retriever.retrieve_top_k(low_id_features)
            retrieval_id_features = retrieval_id_features.to(accelerator.device)
            
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
            retrieval_clip_features_tensor = torch.from_numpy(np.stack(clip_features_list)).to(accelerator.device)
                
            noise_pred, retrieval_tokens, high_id_tokens, low_id_tokens = ip_retrieval(
                noisy_latents=noisy_latents,
                timesteps=timesteps,
                encoder_hidden_states=encoder_hidden_states,
                low_id_features=low_id_features,
                high_id_features=high_id_features,
                retrieval_clip_features=retrieval_clip_features_tensor,
                retrieval_id_features=retrieval_id_features
            )

            noise_loss = F.mse_loss(noise_pred, noise)
            id_align_loss = compute_id_align_loss(low_id_tokens, high_id_tokens)
            retrieval_align_loss = compute_retrieval_align_loss(retrieval_tokens, high_clip_features, high_id_features)

            generated_latents = []
            for i in range(noise_pred.shape[0]):
                step_output = noise_scheduler.step(
                    model_output=noise_pred[i].unsqueeze(0),
                    timestep=timesteps[i].unsqueeze(0),
                    sample=noisy_latents[i].unsqueeze(0)
                )
                generated_latents.append(step_output.pred_original_sample)
            generated_latents = torch.cat(generated_latents, dim=0)
            
            gen_images = vae.decode(generated_latents / vae.config.scaling_factor, return_dict=False)[0]
            gen_images = (gen_images.clamp(-1, 1) + 1) / 2
            real_images = (batch["high_images"].to(accelerator.device).clamp(-1, 1) + 1) / 2
            lpips_loss = lpips_model(gen_images, real_images).mean()

            total_noise_loss += noise_loss.item()
            total_id_align_loss += id_align_loss.item()
            total_retrieval_align_loss += retrieval_align_loss.item()
            total_lpips_loss += lpips_loss.item()
            total_loss += (args.lambda_mse * noise_loss.item() +
                           args.lambda_id * id_align_loss.item() +
                           args.lambda_retrieval * retrieval_align_loss.item() +
                           args.lambda_lpips * lpips_loss.item())
            num_batches += 1

    ip_retrieval.train()
    avg_noise_loss = total_noise_loss / num_batches
    avg_id_align_loss = total_id_align_loss / num_batches
    avg_retrieval_align_loss = total_retrieval_align_loss / num_batches
    avg_lpips_loss = total_lpips_loss / num_batches
    avg_total_loss = total_loss / num_batches
    return avg_noise_loss, avg_id_align_loss, avg_retrieval_align_loss, avg_lpips_loss, avg_total_loss

def main():
    args = parse_args()
    set_seed(args.seed)
    
    accelerator = Accelerator(
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=ProjectConfiguration(project_dir=args.output_dir, logging_dir=Path(args.output_dir, args.logging_dir)),
        kwargs_handlers=[DistributedDataParallelKwargs(find_unused_parameters=True)]
    )
    
    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)

    noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
    tokenizer = CLIPTokenizer.from_pretrained(args.pretrained_model_name_or_path, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="text_encoder")
    vae = AutoencoderKL.from_pretrained(args.pretrained_model_name_or_path, subfolder="vae")
    unet = UNet2DConditionModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="unet")

    vae.to(accelerator.device)
    text_encoder.to(accelerator.device)
    unet.to(accelerator.device)

    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    unet.requires_grad_(False)
    unet.enable_xformers_memory_efficient_attention()
    unet.set_attention_slice("auto")

    retrieval_proj_model = MLPProjRetrieModel(cross_attention_dim=unet.config.cross_attention_dim, embeddings_dim=768, num_tokens=16)
    id_proj_high = MLPProjModel(cross_attention_dim=unet.config.cross_attention_dim, id_embeddings_dim=512, num_tokens=16)
    id_proj_low = MLPProjModel(cross_attention_dim=unet.config.cross_attention_dim, id_embeddings_dim=512, num_tokens=16)
    transformer = RAFAAggregator(d_model=unet.config.cross_attention_dim, nhead=4, dim_feedforward=1024)
    
    retriever = FaissRetriever(
        index_path=os.path.join(args.feature_db_path, "retrieval/face_ivf_index.faiss"),
        features_path=os.path.join(args.feature_db_path, "retrieval/all_features.npy"),
        key_mapping_path=os.path.join(args.feature_db_path, "retrieval/face_key_mapping.json"),
        index_type="ip",
        nlist=100,
        nprobe=10,
        top_k_rough=50,
        top_k_final=args.top_k,
        use_gpu=True
    )
    
    train_dataset = MyDataset(
        args.data_json_file, 
        tokenizer, 
        size=args.resolution, 
        image_root_path=args.data_root_path, 
        feature_db_path=args.feature_db_path, 
        split="train"
    )
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, 
        shuffle=True, 
        collate_fn=collate_fn, 
        batch_size=args.train_batch_size, 
        num_workers=args.dataloader_num_workers, 
        pin_memory=True
    )
    val_dataset = MyDataset(
        args.data_json_file, 
        tokenizer, 
        size=args.resolution, 
        image_root_path=args.data_root_path, 
        feature_db_path=args.feature_db_path, 
        split="valid"
    )
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset, 
        shuffle=False, 
        collate_fn=collate_fn, 
        batch_size=args.train_batch_size, 
        num_workers=args.dataloader_num_workers, 
        pin_memory=True
    )

    attn_procs = {}
    lora_rank = 128
    unet_sd = unet.state_dict()
    for name, module in unet.up_blocks[-1].named_modules():
        for param_name, param in module.named_parameters(recurse=False):
            if 'bias' in param_name or isinstance(module, torch.nn.LayerNorm):
                param.requires_grad = True
            else:
                param.requires_grad = False

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
            attn_procs[name] = LoRAAttnProcessor(hidden_size=hidden_size, cross_attention_dim=cross_attention_dim, rank=lora_rank)
        else:
            layer_name = name.split(".processor")[0]
            weights = {"to_k_ip.weight": unet_sd[layer_name + ".to_k.weight"], "to_v_ip.weight": unet_sd[layer_name + ".to_v.weight"]}
            attn_procs[name] = FaceLoRAAttentionProcessor(
                hidden_size=hidden_size,
                cross_attention_dim=cross_attention_dim,
                rank=lora_rank,
                num_tokens=16,
                num_heads=8,
                use_residual=True,
                dropout_p=0.1
            )
            attn_procs[name].load_state_dict(weights, strict=False)

    unet.set_attn_processor(attn_procs)
    adapter_modules = torch.nn.ModuleList(unet.attn_processors.values())

    ip_retrieval = IPRetrieval(
        unet=unet,
        adapter_modules=adapter_modules,
        retrieval_proj_model=retrieval_proj_model,
        id_proj_high=id_proj_high,
        id_proj_low=id_proj_low,
        transformer=transformer,
        ckpt_path=args.pretrained_ip_adapter_path
    )

    train(args, accelerator, train_dataloader, val_dataloader, ip_retrieval, noise_scheduler, vae, text_encoder, retriever)

if __name__ == "__main__":
    main()