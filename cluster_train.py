import os
import subprocess
import sys

# ==========================================
# 0. 自动寻找空闲 GPU
# ==========================================
def auto_select_gpu():
    try:
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
torch.set_float32_matmul_precision('high')

# ==========================================
# 2. 全局配置与超参数
# ==========================================

# --- 可视化配置 (新增) ---
VISUALIZATION_INTERVALS = [0, 1000, 3000, 5000, 7500, 10000]

# --- 核心重建质量参数 ---
NUM_FREQS = 24           
LATENT_DIM = 64          
NUM_CODES = 256        
NUM_LATENT_VECTORS = 4   

# --- 训练配置 ---
BATCH_SIZE = 150         
EPOCHS = 10000           
LR = 1e-3                
SEED = 42                

# --- 模型架构参数 ---
HIDDEN_SIZE = 128        
NUM_LAYERS = 3           
COORD_DIM = 2            
VALUE_DIM = 3            
COMMITMENT_COST = 0.25   

# --- 数据集配置 ---
IMAGE_SIZE = 64          
NUM_IMAGES_PER_DS = 50   
NUM_WORKERS = 8          

# --- 输出配置 ---
RESULTS_DIR = 'results_evolution'  # 修改输出文件夹名

# ==========================================
# 3. 基础模块 (保持不变)
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
        embedding = torch.randn(num_codes, code_dim)
        embedding = embedding / embedding.norm(dim=1, keepdim=True)
        self.register_buffer("embedding", embedding)
        self.register_buffer("ema_cluster_size", torch.zeros(num_codes))
        self.register_buffer("ema_embedding", self.embedding.clone())

    def forward(self, z):
        original_dtype = z.dtype
        z_fp32 = z.float()
        z_flat = z_fp32.view(-1, self.code_dim)
        distances = (z_flat.pow(2).sum(dim=1, keepdim=True) 
                     - 2 * z_flat @ self.embedding.t() 
                     + self.embedding.pow(2).sum(dim=1))
        indices = torch.argmin(distances, dim=1)
        z_q = F.embedding(indices, self.embedding)
        if self.training and torch.is_grad_enabled():
            encodings = F.one_hot(indices, self.num_codes).float()
            self.ema_cluster_size.mul_(self.decay).add_(encodings.sum(0) * (1 - self.decay))
            dw = encodings.t() @ z_flat.detach()
            self.ema_embedding.mul_(self.decay).add_(dw * (1 - self.decay))
            n = self.ema_cluster_size.sum()
            cluster_size = (self.ema_cluster_size + self.epsilon) / (n + self.num_codes * self.epsilon) * n
            self.embedding.copy_(self.ema_embedding / cluster_size.unsqueeze(1))
        z_q_st = z_fp32 + (z_q - z_fp32).detach()
        vq_loss = self.commitment_cost * F.mse_loss(z_q.detach(), z_fp32)
        return z_q_st.to(original_dtype), indices, vq_loss

# ==========================================
# 4. 核心网络 VQINR
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
        H, W = resolution
        y = torch.linspace(-1, 1, H, device=device)
        x = torch.linspace(-1, 1, W, device=device)
        yy, xx = torch.meshgrid(y, x, indexing='ij')
        coords = torch.stack([yy, xx], dim=-1).reshape(1, -1, 2)
        indices = torch.tensor([latent_idx], device=device)
        pred, _, _ = self(coords, indices)
        return pred.reshape(H, W, -1)

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
# 6. Lightning Module (增强版：支持历史记录)
# ==========================================

class MultiImageINRModule(pl.LightningModule):
    def __init__(self, network, gt_images, track_indices, vis_intervals, lr):
        super().__init__()
        self.network = network
        self.lr = lr
        # 将 GT 转到 CPU
        self.gt_images = [t.detach().cpu() for t in gt_images]
        
        # 可视化相关
        self.vis_intervals = set(vis_intervals)
        # track_indices: {'breast': [0,1,2], 'retina': [50,51,52], ...}
        self.track_indices = track_indices 
        # 存储历史记录: {epoch: {global_idx: image_tensor}}
        self.history = {} 

    def configure_optimizers(self):
        return torch.optim.Adam(self.network.parameters(), lr=self.lr)
    
    def training_step(self, batch, batch_idx):
        coords, values, img_indices, masks = batch
        outputs, _, vq_loss = self.network(coords, latent_indices=img_indices)
        masks_expanded = masks.unsqueeze(1).expand_as(outputs)
        loss_sq = (outputs - values) ** 2
        recon_loss = (loss_sq * masks_expanded).mean()
        loss = recon_loss + vq_loss
        self.log('train/recon_loss', recon_loss, prog_bar=True)
        self.log('train/vq_loss', vq_loss, prog_bar=True)
        return loss

    @torch.no_grad()
    def save_snapshots(self, current_epoch):
        """保存当前时刻的追踪样本快照"""
        if current_epoch not in self.vis_intervals:
            return
            
        print(f"\n[Epoch {current_epoch}] Saving visualization snapshots...")
        epoch_snapshots = {}
        
        # 遍历所有需要追踪的数据集和索引
        for ds_name, indices in self.track_indices.items():
            for idx in indices:
                gt = self.gt_images[idx].to(self.device)
                H, W, C_orig = gt.shape
                
                # 重建
                pred_full = self.network.get_image((H, W), idx, self.device)
                
                # 处理通道和截断
                if C_orig == 1:
                    pred = pred_full[..., 0:1] # 灰度
                else:
                    pred = pred_full # 彩色
                
                # 转回 CPU 存储以节省显存
                epoch_snapshots[idx] = torch.clamp(pred, 0, 1).cpu()
        
        self.history[current_epoch] = epoch_snapshots

    def on_fit_start(self):
        # 记录 Epoch 0 (未训练/初始化状态)
        if 0 in self.vis_intervals:
            self.save_snapshots(0)

    def on_train_epoch_end(self):
        # 每个 Epoch 结束时的标准 PSNR 日志
        idx = list(self.track_indices.values())[0][0] # 随便取第一个样本
        gt = self.gt_images[idx].to(self.device)
        H, W, C_orig = gt.shape
        pred_full = self.network.get_image((H, W), idx, self.device)
        score = psnr(pred_full[..., :C_orig], gt)
        self.log('val/psnr', score, prog_bar=True)
        
        # 检查是否是可视化节点 (Epoch 计数从 0 开始，这里用 current_epoch + 1 代表训练完的轮数)
        # 例如 1000 轮训练完，current_epoch 是 999，+1 = 1000
        epoch_num = self.current_epoch + 1
        if epoch_num in self.vis_intervals:
            self.save_snapshots(epoch_num)

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
    
    # 用于记录哪些索引属于哪个数据集，以便后续可视化
    # 格式: {'breast': [0, 1, 2], 'retina': [50, 51, 52]...}
    track_indices = {} 
    current_idx_offset = 0
    
    print(f"Sampling {NUM_IMAGES_PER_DS} images from each dataset...")
    for ds_name, ds in datasets.items():
        count = min(len(ds), NUM_IMAGES_PER_DS)
        indices = random.sample(range(len(ds)), count)
        
        # 记录前 3 个样本用于可视化追踪
        dataset_global_indices = []
        
        for i, sample_idx in enumerate(indices):
            img = pil_to_tensor(ds[sample_idx][0]).moveaxis(0, -1).float() / 255.0
            all_images_original.append(img)
            
            # 记录全局索引
            if i < 3: # 每个数据集选 3 个
                dataset_global_indices.append(current_idx_offset + i)
        
        track_indices[ds_name] = dataset_global_indices
        current_idx_offset += len(indices)
            
    print(f"Total training images: {len(all_images_original)}")
    print(f"Tracking indices for visualization: {track_indices}")
    
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
    
    with torch.no_grad():
        for m in vqinr.decoder.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None: m.bias.data.fill_(0)
    
    print("Compiling model for H100 acceleration...")
    vqinr.decoder = torch.compile(vqinr.decoder, mode="reduce-overhead")
    
    # 传入需要追踪的索引和时间点
    module = MultiImageINRModule(vqinr, all_images_original, track_indices, VISUALIZATION_INTERVALS, lr=LR)
    
    trainer = pl.Trainer(
        max_epochs=EPOCHS,
        accelerator="gpu",
        devices=1,           
        precision="bf16-mixed",
        enable_checkpointing=False,
        logger=False,
        log_every_n_steps=10
    )
    
    print("\nStarting Training...")
    trainer.fit(module, dataloader)
    
    # --- 训练后生成演变图 ---
    print("\nGenerating Evolution Plots...")
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    # 获取排序后的时间点，确保画图顺序正确
    sorted_intervals = sorted([t for t in VISUALIZATION_INTERVALS if t in module.history])
    
    # 为每个 Dataset 生成一张大图
    for ds_name, indices in track_indices.items():
        print(f"Plotting evolution for {ds_name}...")
        
        # 3 行 (3个样本)，列数 = 1 (GT) + 时间点数量
        rows = len(indices)
        cols = 1 + len(sorted_intervals)
        
        fig, axes = plt.subplots(rows, cols, figsize=(3 * cols, 3 * rows))
        
        # 如果只有一行，axes 是一维数组，需要转二维方便索引
        if rows == 1: axes = axes[None, :]
        
        for r, global_idx in enumerate(indices):
            # 1. 画 Ground Truth
            gt = all_images_original[global_idx]
            if gt.shape[-1] == 1:
                axes[r, 0].imshow(gt.squeeze(), cmap='gray')
            else:
                axes[r, 0].imshow(gt)
            axes[r, 0].axis('off')
            if r == 0: axes[r, 0].set_title("Ground Truth", fontsize=10, fontweight='bold')
            
            # 2. 画各个时间点的重建图
            for c, epoch in enumerate(sorted_intervals):
                # 从历史记录中获取图片
                if global_idx in module.history[epoch]:
                    recon_img = module.history[epoch][global_idx]
                    
                    if recon_img.shape[-1] == 1:
                        axes[r, c+1].imshow(recon_img.squeeze(), cmap='gray')
                    else:
                        axes[r, c+1].imshow(recon_img)
                    
                    # 计算当前 PSNR
                    curr_psnr = psnr(recon_img, gt)
                    axes[r, c+1].set_title(f"Step {epoch}\n{curr_psnr:.1f} dB", fontsize=9)
                else:
                    axes[r, c+1].text(0.5, 0.5, "Missing", ha='center')
                
                axes[r, c+1].axis('off')

        plt.tight_layout()
        save_path = os.path.join(RESULTS_DIR, f'evolution_{ds_name}.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved {save_path}")

    print(f"\n✅ All evolution plots saved to {RESULTS_DIR}")