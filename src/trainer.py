"""PyTorch Lightning training module for VQINR"""

import torch
import lightning as pl
from src.utils import psnr_from_images


def gradient_loss_masked(pred, gt, masks):
    """
    Compute gradient loss with channel masking
    Args:
        pred: predicted images (B, H, W, C)
        gt: ground truth images (B, H, W, C)
        masks: channel masks (B, C) in {0,1}
    Returns:
        gradient loss (scalar)
    """
    B, H, W, C = pred.shape
    m = masks.view(B, 1, 1, C)

    dx_pred = pred[:, 1:, :, :] - pred[:, :-1, :, :]
    dx_gt = gt[:, 1:, :, :] - gt[:, :-1, :, :]

    dy_pred = pred[:, :, 1:, :] - pred[:, :, :-1, :]
    dy_gt = gt[:, :, 1:, :] - gt[:, :, :-1, :]

    loss = ((dx_pred - dx_gt).abs() * m).mean() + ((dy_pred - dy_gt).abs() * m).mean()
    return loss


class MultiImageINRModule(pl.LightningModule):
    """Lightning module for training multi-image INR with VQ"""

    def __init__(
        self,
        network,
        gt_images,
        track_indices,
        vis_intervals,
        lr,
        grad_loss_weight=0.0,
        log_every_epoch=False,
    ):
        """
        Args:
            network: VQINR model
            gt_images: list of ground truth images
            track_indices: dict mapping dataset names to image indices for tracking
            vis_intervals: set of epochs at which to save visualizations
            lr: learning rate
            grad_loss_weight: weight for gradient loss (default: 0.0)
            log_every_epoch: whether to print epoch summaries every epoch
        """
        super().__init__()
        self.network = network
        self.lr = lr
        self.grad_loss_weight = grad_loss_weight
        self.gt_images = [t.detach().cpu() for t in gt_images]
        self.vis_intervals = set(vis_intervals)
        self.track_indices = track_indices
        self.log_every_epoch = log_every_epoch
        self.history = {}
        self.last_rec_loss = None

        # Initialize history tracking
        keys = ["total"] + list(track_indices.keys())
        self.psnr_history = {k: [] for k in keys}
        self.codebook_usage_history = []

    def configure_optimizers(self):
        """Configure optimizer"""
        return torch.optim.Adam(self.network.parameters(), lr=self.lr)

    def training_step(self, batch, batch_idx):
        """Training step"""
        self.network.current_epoch = self.current_epoch
        coords, values, img_indices, masks = batch
        outputs, _, vq_loss = self.network(coords, latent_indices=img_indices)

        # Masked reconstruction loss
        masks_expanded = masks.unsqueeze(1).expand_as(outputs)
        loss_sq = (outputs - values) ** 2
        recon_loss = (loss_sq * masks_expanded).mean()

        loss = recon_loss + vq_loss

        # Add gradient loss if weight > 0
        if self.grad_loss_weight > 0:
            # Get image resolution from first image
            B = coords.shape[0]
            # Assumes square images; non-square grids would need explicit H/W.
            H = W = int(coords.shape[1] ** 0.5)

            # Reshape outputs and values to image format
            pred_imgs = outputs.view(B, H, W, -1)
            gt_imgs = values.view(B, H, W, -1)

            grad_loss = gradient_loss_masked(pred_imgs, gt_imgs, masks)

            # Cosine warmup for gradient loss
            warmup_epochs = 3000
            if self.current_epoch < warmup_epochs:
                import math

                warmup_scale = 0.5 * (
                    1 - math.cos(math.pi * self.current_epoch / warmup_epochs)
                )
            else:
                warmup_scale = 1.0

            loss = loss + self.grad_loss_weight * warmup_scale * grad_loss

            self.log("train/grad_loss", grad_loss, prog_bar=False)
            self.log(
                "train/grad_loss_weight",
                warmup_scale * self.grad_loss_weight,
                prog_bar=False,
            )

        self.last_rec_loss = recon_loss.detach()
        self.log("train/loss", loss, prog_bar=True)
        return loss

    @torch.no_grad()
    def save_snapshots(self, current_epoch):
        """Save visualization snapshots at specified epochs"""
        if current_epoch not in self.vis_intervals:
            return

        print(f"\n[Epoch {current_epoch}] Saving visualization snapshots...")
        epoch_snapshots = {}

        for ds_name, indices in self.track_indices.items():
            for idx in indices:
                gt = self.gt_images[idx].to(self.device)
                H, W, C_orig = gt.shape
                pred_full = self.network.get_image((H, W), idx, self.device)
                pred = pred_full[..., 0:1] if C_orig == 1 else pred_full
                epoch_snapshots[idx] = torch.clamp(pred, 0, 1).cpu()

        self.history[current_epoch] = epoch_snapshots

    def on_fit_start(self):
        """Called at the start of training"""
        if 0 in self.vis_intervals:
            self.save_snapshots(0)

    def on_train_epoch_end(self):
        """Called at the end of each training epoch"""
        epoch_num = self.current_epoch + 1

        # Save snapshots if needed
        if epoch_num in self.vis_intervals:
            self.save_snapshots(epoch_num)

        # Track codebook utilization
        total_codes = 0
        used_codes = 0
        for vq_layer in self.network.vq_layers:
            used = (vq_layer.ema_cluster_size > 0.01).sum().item()
            used_codes += used
            total_codes += vq_layer.num_codes

        utilization = (used_codes / total_codes) * 100.0 if total_codes > 0 else 0
        self.codebook_usage_history.append(utilization)

        # Compute PSNR for tracked images
        total_psnr_sum = 0
        total_count = 0

        for ds_name, indices in self.track_indices.items():
            if len(indices) == 0:
                continue
            ds_psnr_sum = 0
            for idx in indices:
                gt = self.gt_images[idx].to(self.device)
                H, W, C_orig = gt.shape
                pred_full = self.network.get_image((H, W), idx, self.device)
                val = psnr_from_images(pred_full[..., :C_orig], gt)
                ds_psnr_sum += val

            avg_ds_psnr = ds_psnr_sum / len(indices)
            self.psnr_history[ds_name].append(avg_ds_psnr.cpu().item())

            total_psnr_sum += ds_psnr_sum
            total_count += len(indices)

        avg_total_psnr = (
            total_psnr_sum / total_count
            if total_count > 0
            else torch.tensor(0.0, device=self.device)
        )
        self.psnr_history["total"].append(avg_total_psnr.cpu().item())

        self.log("val/psnr", avg_total_psnr, prog_bar=True)
        self.log("val/codebook", utilization, prog_bar=True)

        if self.log_every_epoch or (epoch_num in self.vis_intervals):
            psnr_display = (
                f"{avg_total_psnr.item():.2f} dB" if total_count > 0 else "n/a"
            )
            if self.last_rec_loss is not None:
                print(
                    f"Epoch {epoch_num} | Rec loss: {self.last_rec_loss.item():.6f} | "
                    f"Avg PSNR: {psnr_display}"
                )
