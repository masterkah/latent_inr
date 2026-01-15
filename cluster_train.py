import os
import subprocess
import sys

# ==========================================
# 0. 自动寻找空闲 GPU (必须放在所有 Torch 操作之前)
# ==========================================
def auto_select_gpu():
    try:
        # 如果用户已经手动指定了，就不自动选了
        if "CUDA_VISIBLE_DEVICES" in os.environ:
            print(f"Using manually set CUDA_VISIBLE_DEVICES={os.environ['CUDA_VISIBLE_DEVICES']}")
            return

        cmd = "nvidia-smi --query-gpu=index,memory.free,utilization.gpu --format=csv,nounits,noheader"
        result = subprocess.check_output(cmd.split(), encoding='utf-8')
        
        lines = result.strip().split('\n')
        best_gpu = -1
        max_free = 0
        
        print("Scanning GPUs...")
        for line in lines:
            try:
                idx, free_mem, util = line.split(', ')
                idx, free_mem, util = int(idx), int(free_mem), int(util)
                print(f"  GPU {idx}: Free Memory: {free_mem}MiB, Utilization: {util}%")
                
                # 策略：优先找利用率 < 5% 且显存剩余最大的
                if util < 5 and free_mem > max_free:
                    max_free = free_mem
                    best_gpu = idx
            except:
                continue
        
        if best_gpu != -1:
            print(f"\n✅ Auto-selected GPU {best_gpu} (Free: {max_free}MiB)")
            os.environ["CUDA_VISIBLE_DEVICES"] = str(best_gpu)
        else:
            print("⚠️ No ideal GPU found, using default strategy.")
            
    except Exception as e:
        print(f"⚠️ GPU auto-selection failed: {e}")

auto_select_gpu()

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import lightning as pl
import numpy as np
import random
import matplotlib.pyplot as plt
import math
from torchvision.transforms.functional import pil_to_tensor
from medmnist import BreastMNIST, RetinaMNIST, PneumoniaMNIST

# ==========================================
# 1. 硬件加速设置
# ==========================================
# 开启 TensorFloat-32
torch.set_float32_matmul_precision('high')

# ==========================================
# 2. 全局配置与超参数
# ==========================================

# --- 核心重建质量参数 ---
NUM_FREQS = 24           # 频率数
LATENT_DIM = 64          # 潜向量维度
NUM_CODES = 1024         # 码本大小
NUM_LATENT_VECTORS = 4   # Residual VQ 级数

# --- 训练配置 ---
BATCH_SIZE = 150         # Full Batch
EPOCHS = 10000           
LR = 1e-3                
SEED = 42                

# --- 模型架构参数 ---
HIDDEN_SIZE = 256        
NUM_LAYERS = 4           
COORD_DIM = 2            
VALUE_DIM = 3            
COMMITMENT_COST = 0.25   

# --- 数据集配置 ---
IMAGE_SIZE = 64          
NUM_IMAGES_PER_DS = 50   
NUM_WORKERS = 8          

# --- 输出配置 ---
RESULTS_DIR = 'results_h100_fixed'

# ==========================================
# 3. 基础模块 (修复数值稳定性)
# ==========================================

def psnr(pred, target):
    mse = torch.mean((pred - target) ** 2)
    if mse == 0: return float('inf')
    return 20 * torch.log10(1.0 / torch.sqrt(mse))

class PositionalEncoding(nn.Module):
    def __init__(self, num_freqs: int, include_input: bool = True):
        super().__init__()
        self.num_freqs = num_freqs
        self.include_input = include_input
        freqs = 2.0 ** torch.arange(num_freqs)
        self.register_buffer('freqs', freqs)

    def forward(self, x):
        embed = [x] if self.include_input else []
        x_expanded = x.unsqueeze(-1) * self.freqs * torch.pi
        sin_x = torch.sin(x_expanded)
        cos_x = torch.cos(x_expanded)
        embed.append(sin_x.reshape(x.shape[0], -1))
        embed.append(cos_x.reshape(x.shape[0], -1))
        return torch.cat(embed, dim=-1)

class ShiftReLULayer(nn.Module):
    def __init__(self, in_size, out_size):
        super().__init__()
        self.linear = nn.Linear(in_size, out_size)

    def forward(self, x, beta=None):
        out = self.linear(x)
        if beta is not None:
            out = out + beta
        return torch.relu(out)

class ModulatedReLU(nn.Module):
    def __init__(self, in_size, out_size, hidden_size, num_layers):
        super().__init__()
        self.layers = nn.ModuleList()
        self.layers.append(ShiftReLULayer(in_size, hidden_size))
        for _ in range(num_layers - 1):
            self.layers.append(ShiftReLULayer(hidden_size, hidden_size))
        self.last_layer = nn.Linear(hidden_size, out_size)

    def forward(self, x, betas=None):
        for i, layer in enumerate(self.layers):
            beta = betas[i] if betas is not None else None
            x = layer(x, beta=beta)
        x = self.last_layer(x)
        return x

class EMAVectorQuantizer(nn.Module):
    def __init__(self, num_codes, code_dim, decay=0.99, epsilon=1e-5, commitment_cost=0.25):
        super().__init__()
        self.num_codes = num_codes
        self.code_dim = code_dim
        self.decay = decay
        self.epsilon = epsilon
        self.commitment_cost = commitment_cost

        # 初始化为 float32
        embedding = torch.randn(num_codes, code_dim)
        embedding = embedding / embedding.norm(dim=1, keepdim=True)
        self.register_buffer("embedding", embedding)
        self.register_buffer("ema_cluster_size", torch.zeros(num_codes))
        self.register_buffer("ema_embedding", self.embedding.clone())

    def forward(self, z):
        # [Fix 1] 强制转换为 float32 进行距离计算和更新
        # 防止 bf16 下的数值抖动和 NaN
        original_dtype = z.dtype
        z_fp32 = z.float()
        
        z_flat = z_fp32.view(-1, self.code_dim)
        
        # 距离计算 (FP32)
        distances = (z_flat.pow(2).sum(dim=1, keepdim=True) 
                     - 2 * z_flat @ self.embedding.t() 
                     + self.embedding.pow(2).sum(dim=1))
        
        indices = torch.argmin(distances, dim=1)
        z_q = F.embedding(indices, self.embedding) # Embedding buffer 也是 fp32
        
        if self.training and torch.is_grad_enabled():
            encodings = F.one_hot(indices, self.num_codes).float()
            # EMA 更新 (FP32)
            self.ema_cluster_size.mul_(self.decay).add_(encodings.sum(0) * (1 - self.decay))
            dw = encodings.t() @ z_flat.detach()
            self.ema_embedding.mul_(self.decay).add_(dw * (1 - self.decay))
            
            n = self.ema_cluster_size.sum()
            cluster_size = (self.ema_cluster_size + self.epsilon) / (n + self.num_codes * self.epsilon) * n
            self.embedding.copy_(self.ema_embedding / cluster_size.unsqueeze(1))
            
        # Straight-through estimator
        z_q_st = z_fp32 + (z_q - z_fp32).detach()
        
        vq_loss = self.commitment_cost * F.mse_loss(z_q.detach(), z_fp32)
        
        # [Fix 1 End] 转回原始精度 (如 bf16) 以兼容后续网络
        return z_q_st.to(original_dtype), indices, vq_loss

# ==========================================
# 4. 核心网络 VQINR (修复 Loss 和推理)
# ==========================================

class VQINR(nn.Module):
    def __init__(self, coord_dim, value_dim, latent_dim, num_codes, 
                 hidden_size, num_layers, num_latent_vectors, 
                 num_images, num_freqs, commitment_cost, **kwargs):
        super().__init__()
        
        self.coord_dim = coord_dim
        self.latent_dim = latent_dim
        self.num_latent_vectors = num_latent_vectors
        
        self.latents = nn.Parameter(torch.randn(num_images, num_latent_vectors, latent_dim))
        self.vq_layers = nn.ModuleList([
            EMAVectorQuantizer(num_codes=num_codes, code_dim=latent_dim, commitment_cost=commitment_cost)
            for _ in range(num_latent_vectors)
        ])
        
        self.modulation_layers = nn.ModuleList([
            nn.Linear(latent_dim, hidden_size) for _ in range(num_layers)
        ])
        
        self.pos_enc = PositionalEncoding(num_freqs)
        decoder_in_dim = coord_dim + coord_dim * 2 * num_freqs
        
        self.decoder = ModulatedReLU(decoder_in_dim, value_dim, hidden_size, num_layers)

    def forward(self, coords, latent_indices):
        batch_size = coords.shape[0]
        num_points = coords.shape[1]
        
        img_latents = self.latents[latent_indices] 
        
        z_q_sum = torch.zeros(batch_size, self.latent_dim, device=coords.device, dtype=img_latents.dtype)
        residual_target = torch.zeros(batch_size, self.latent_dim, device=coords.device, dtype=img_latents.dtype)
        total_vq_loss = 0.0
        
        for stage_idx in range(self.num_latent_vectors):
            z_stage = img_latents[:, stage_idx, :]
            
            if stage_idx == 0:
                residual_target = z_stage
                z_current = z_stage
            else:
                residual_target = residual_target + z_stage
                z_current = residual_target - z_q_sum.detach()
            
            z_q_stage, _, vq_loss = self.vq_layers[stage_idx](z_current)
            z_q_sum = z_q_sum + z_q_stage
            total_vq_loss += vq_loss
        
        # [Fix 2] VQ Loss 归一化：除以阶段数
        total_vq_loss = total_vq_loss / self.num_latent_vectors
            
        betas = [layer(z_q_sum) for layer in self.modulation_layers]
        
        betas_expanded = []
        for beta in betas:
            beta_exp = beta.unsqueeze(1).expand(-1, num_points, -1).reshape(-1, beta.shape[-1])
            betas_expanded.append(beta_exp)
            
        coords_flat = coords.reshape(-1, self.coord_dim)
        coords_enc = self.pos_enc(coords_flat)
        values_flat = self.decoder(coords_enc, betas=betas_expanded)
        
        return values_flat.view(batch_size, num_points, -1), None, total_vq_loss

    @torch.no_grad()
    def get_image(self, resolution, latent_idx, device):
        # 这里的 forward 包含了 VQ 过程，用于验证
        H, W = resolution
        y = torch.linspace(-1, 1, H, device=device)
        x = torch.linspace(-1, 1, W, device=device)
        yy, xx = torch.meshgrid(y, x, indexing='ij')
        coords = torch.stack([yy, xx], dim=-1).reshape(1, -1, 2)
        indices = torch.tensor([latent_idx], device=device)
        pred, _, _ = self(coords, indices)
        return pred.reshape(H, W, -1)
    
    @torch.no_grad()
    def decode_from_indices(self, resolution, indices_list, device):
        # [Fix 3] 纯索引解码路径 (模拟真实压缩解压)
        # indices_list: List of tensor(1,) [idx_stage0, idx_stage1, ...]
        H, W = resolution
        y = torch.linspace(-1, 1, H, device=device)
        x = torch.linspace(-1, 1, W, device=device)
        yy, xx = torch.meshgrid(y, x, indexing='ij')
        coords_flat = torch.stack([yy, xx], dim=-1).reshape(-1, 2)
        
        # 重建 Latent
        z_q_sum = torch.zeros(1, self.latent_dim, device=device)
        for stage_idx, idx in enumerate(indices_list):
            z_q = F.embedding(idx, self.vq_layers[stage_idx].embedding)
            z_q_sum = z_q_sum + z_q
            
        # Modulation
        betas = [layer(z_q_sum) for layer in self.modulation_layers]
        
        # Expand Betas
        num_points = H * W
        betas_expanded = []
        for beta in betas:
             # beta: (1, dim) -> (points, dim)
             betas_expanded.append(beta.expand(num_points, -1))
             
        # Decode
        coords_enc = self.pos_enc(coords_flat)
        values_flat = self.decoder(coords_enc, betas=betas_expanded)
        
        return values_flat.reshape(H, W, -1)

# ==========================================
# 5. 数据集
# ==========================================

class FullImageDataset(Dataset):
    def __init__(self, images: list):
        super().__init__()
        self.images = []
        self.masks = []
        print("Preprocessing images to RAM (Full Grid)...")
        for img in images:
            h, w, c = img.shape
            if c == 1:
                zeros = torch.zeros(h, w, 2, dtype=img.dtype)
                img_padded = torch.cat([img, zeros], dim=-1)
                mask = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float32)
            elif c == 3:
                img_padded = img
                mask = torch.tensor([1.0, 1.0, 1.0], dtype=torch.float32)
            
            y = torch.linspace(-1, 1, h)
            x = torch.linspace(-1, 1, w)
            yy, xx = torch.meshgrid(y, x, indexing='ij')
            coords = torch.stack([yy, xx], dim=-1).reshape(-1, 2)
            values = img_padded.reshape(-1, 3)
            
            self.images.append((coords, values))
            self.masks.append(mask)

    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx: int):
        return self.images[idx][0], self.images[idx][1], idx, self.masks[idx]

# ==========================================
# 6. Lightning Module
# ==========================================

class MultiImageINRModule(pl.LightningModule):
    def __init__(self, network, gt_images, lr):
        super().__init__()
        self.network = network
        self.lr = lr
        self.gt_images = [t.detach().cpu() for t in gt_images]
        
    def configure_optimizers(self):
        return torch.optim.Adam(self.network.parameters(), lr=self.lr)
    
    def training_step(self, batch, batch_idx):
        coords, values, img_indices, masks = batch
        outputs, _, vq_loss = self.network(coords, latent_indices=img_indices)
        
        masks_expanded = masks.unsqueeze(1).expand_as(outputs)
        loss_sq = (outputs - values) ** 2
        recon_loss = (loss_sq * masks_expanded).mean()
        
        loss = recon_loss + vq_loss
        
        self.log('train/loss', loss, prog_bar=True)
        self.log('train/recon_loss', recon_loss, prog_bar=True)
        self.log('train/vq_loss', vq_loss, prog_bar=True)
        return loss

    def on_train_epoch_end(self):
        # 简单的验证逻辑：第一张图 PSNR
        idx = 0
        gt = self.gt_images[idx].to(self.device)
        H, W, C_orig = gt.shape
        pred_full = self.network.get_image((H, W), idx, self.device)
        pred = pred_full[..., :C_orig]
        score = psnr(pred, gt)
        print(f"\n[Epoch {self.current_epoch}] Img 0 PSNR: {score:.2f} dB")
        self.log('val/psnr', score, prog_bar=True)

# ==========================================
# 7. Main
# ==========================================

if __name__ == "__main__":
    pl.seed_everything(SEED)
    
    print("Loading datasets...")
    datasets = {
        'breast': BreastMNIST(split="val", download=True, size=IMAGE_SIZE),
        'retina': RetinaMNIST(split="val", download=True, size=IMAGE_SIZE),
        'pneumonia': PneumoniaMNIST(split="val", download=True, size=IMAGE_SIZE)
    }
    
    all_images_original = [] 
    
    print(f"Sampling {NUM_IMAGES_PER_DS} images from each dataset...")
    for ds_name, ds in datasets.items():
        count = min(len(ds), NUM_IMAGES_PER_DS)
        indices = random.sample(range(len(ds)), count)
        for i in indices:
            img = pil_to_tensor(ds[i][0]).moveaxis(0, -1).float() / 255.0
            all_images_original.append(img)
            
    print(f"Total training images: {len(all_images_original)}")
    
    dataset = FullImageDataset(all_images_original)
    dataloader = DataLoader(
        dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True, 
        num_workers=NUM_WORKERS, 
        pin_memory=True,
        drop_last=False
    )
    
    vqinr = VQINR(
        coord_dim=COORD_DIM, 
        value_dim=VALUE_DIM,
        latent_dim=LATENT_DIM,
        num_codes=NUM_CODES,
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        num_latent_vectors=NUM_LATENT_VECTORS,
        num_images=len(all_images_original),
        num_freqs=NUM_FREQS, 
        commitment_cost=COMMITMENT_COST
    )
    
    # 权重初始化
    with torch.no_grad():
        for m in vqinr.decoder.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None: m.bias.data.fill_(0)
    
    print("Compiling model for H100 acceleration...")
    # 只编译 Decoder 保证最大兼容性
    vqinr.decoder = torch.compile(vqinr.decoder, mode="reduce-overhead")
    
    module = MultiImageINRModule(vqinr, all_images_original, lr=LR)
    
    trainer = pl.Trainer(
        max_epochs=EPOCHS,
        accelerator="gpu",
        devices=1,           
        precision="bf16-mixed",
        enable_checkpointing=False,
        logger=False,
        log_every_n_steps=10
    )
    
    print("\nStarting Training (H100 Optimized + Fixes)...")
    print(f"Hyperparameters: Batch={BATCH_SIZE}, Freqs={NUM_FREQS}, Hidden={HIDDEN_SIZE}, Codes={NUM_CODES}")
    trainer.fit(module, dataloader)
    
    # --- 保存结果 ---
    print("\nTraining done. Saving results...")
    os.makedirs(RESULTS_DIR, exist_ok=True)
    vqinr.eval()
    
    with torch.no_grad():
        fig, axes = plt.subplots(2, 10, figsize=(20, 5))
        for i in range(10):
            if i >= len(all_images_original): break
            gt = all_images_original[i]
            if gt.shape[-1] == 1:
                axes[0, i].imshow(gt.squeeze(), cmap='gray')
            else:
                axes[0, i].imshow(gt)
            axes[0, i].axis('off')
            axes[0, i].set_title("GT")
            
            # 使用 get_image 验证
            pred_full = vqinr.get_image(gt.shape[:-1], i, module.device)
            
            if gt.shape[-1] == 1:
                pred = pred_full[..., 0:1]
                axes[1, i].imshow(pred.cpu().squeeze(), cmap='gray')
            else:
                pred = pred_full
                axes[1, i].imshow(torch.clamp(pred.cpu(), 0, 1))
            axes[1, i].axis('off')
            axes[1, i].set_title(f"{psnr(pred.cpu(), gt):.2f}dB")
            
        plt.tight_layout()
        plt.savefig(f'{RESULTS_DIR}/comparison_fixed.png')
        print(f"Saved comparison to {RESULTS_DIR}/comparison_fixed.png")