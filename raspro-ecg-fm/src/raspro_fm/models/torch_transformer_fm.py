"""Phase 4 Multi-Layer PyTorch Transformer Foundation Models (Apple Silicon MPS / CUDA GPU).

Implements:
  1. RASPROTransformerFM: Hybrid Symbolic-Continuous Multi-Layer Transformer Encoder with:
       - 9-Slot Factorized Clinical Symbol Embeddings + 72-D Slot-and-Pair Prior Projection
       - Continuous 9-Parameter Physiological Residual Projection
       - CMI-Biased Multi-Head Self-Attention (prioritizing high-acuity CAM windows)
       - Multi-Task Self-Supervised Pre-Training:
           (a) CMI-Weighted Masked Slot Modeling (9-slot 5-class Cross-Entropy)
           (b) Masked Continuous Physiological Parameter Reconstruction (Smooth-L1 / Huber)
           (c) Stochastic Sub-Window Contrastive Alignment (InfoNCE)
  2. DeepPatchTransformerFM: ST-MEM / PatchTST-style Multi-Layer PyTorch Masked Autoencoder (MAE)
     operating on L=25 continuous 12-lead ECG patches for apples-to-apples GPU/MPS speed,
     memory, and accuracy comparison.
"""

from dataclasses import dataclass
import math
import time
from typing import Dict, List, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


SYMBOL_TO_IDX: Dict[str, int] = {"D": 0, "L": 1, "N": 2, "H": 3, "E": 4}
RHYTHM_TO_IDX: Dict[str, int] = {"REG": 0, "ECT": 1, "IRR": 2}


def get_Compute_device(prefer_device: str = "mps") -> torch.device:
    """Return best available PyTorch accelerator ('mps', 'cuda', or 'cpu')."""
    if prefer_device == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    if prefer_device == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


@dataclass
class TorchPretrainStats:
    """Wall-clock timing, throughput, and loss metrics from PyTorch Transformer pre-training."""
    device_name: str
    n_records: int
    n_epochs: int
    avg_seq_len: float
    wall_time_sec: float
    ms_per_epoch: float
    records_per_sec: float
    initial_loss: float
    final_loss: float


class CMIBiasedTransformerLayer(nn.Module):
    """Pre-LayerNorm Transformer Encoder layer with CMI-Biased Multi-Head Self-Attention."""

    def __init__(self, d_model: int = 192, n_heads: int = 6, d_ffn: int = 512, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.scale = 1.0 / math.sqrt(self.d_head)

        self.norm1 = nn.LayerNorm(d_model)
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        # Learnable per-head CMI salience bias (initialized to positive clinical prior)
        self.cmi_head_scale = nn.Parameter(torch.linspace(0.5, 2.0, steps=n_heads))
        self.attn_drop = nn.Dropout(dropout)

        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ffn),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ffn, d_model),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        x: torch.Tensor,
        pad_mask: torch.Tensor,
        cmi_weights: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            x: (B, L, d_model)
            pad_mask: (B, L) boolean tensor where True = valid token, False = padding
            cmi_weights: (B, L) float tensor in [0, 1]
        """
        B, L, D = x.shape
        h = self.norm1(x)
        qkv = self.qkv(h).reshape(B, L, 3, self.n_heads, self.d_head).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # (B, H, L, d_head)

        attn_logits = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # (B, H, L, L)
        # Add CMI salience bias to key positions: (B, 1, 1, L) * (1, H, 1, 1)
        cmi_bias = cmi_weights.unsqueeze(1).unsqueeze(2) * self.cmi_head_scale.view(1, self.n_heads, 1, 1)
        attn_logits = attn_logits + cmi_bias

        # Mask out padding keys
        key_mask = (~pad_mask).unsqueeze(1).unsqueeze(2)  # (B, 1, 1, L)
        attn_logits = attn_logits.masked_fill(key_mask, -1e4)

        attn_probs = F.softmax(attn_logits, dim=-1)
        attn_probs = self.attn_drop(attn_probs)
        ctx = torch.matmul(attn_probs, v).transpose(1, 2).reshape(B, L, D)
        x = x + self.out_proj(ctx)
        x = x + self.ffn(self.norm2(x))
        return x


class RASPROTransformerBackbone(nn.Module):
    """Multi-Layer Hybrid Symbolic-Continuous RASPRO Transformer Encoder."""

    def __init__(
        self,
        n_slots: int = 9,
        d_prior: int = 72,
        d_model: int = 192,
        n_heads: int = 6,
        n_layers: int = 4,
        max_len: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.n_slots = n_slots
        self.d_prior = d_prior
        self.d_model = d_model
        self.max_len = max_len

        # 1. Factorized 9-Slot discrete symbol embeddings (0..4 = D,L,N,H,E, 5 = MASK, 6 = PAD)
        self.slot_embs = nn.ModuleList([nn.Embedding(7, d_model) for _ in range(n_slots)])
        # 2. 72-D Compositional Slot-and-Pair Prior projection
        self.prior_proj = nn.Linear(d_prior, d_model)
        # 3. Continuous 9-parameter window feature residual projection
        self.cont_proj = nn.Sequential(
            nn.Linear(n_slots, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        # 4. Rhythm tag (REG/ECT/IRR), CAM indicator, and run-length + positional embeddings
        self.rhythm_emb = nn.Embedding(4, d_model)
        self.run_emb = nn.Embedding(6, d_model)
        self.pos_emb = nn.Embedding(max_len, d_model)
        self.in_norm = nn.LayerNorm(d_model)

        self.layers = nn.ModuleList([
            CMIBiasedTransformerLayer(d_model=d_model, n_heads=n_heads, d_ffn=d_model * 3, dropout=dropout)
            for _ in range(n_layers)
        ])
        self.out_norm = nn.LayerNorm(d_model)
        self.attn_pool_query = nn.Parameter(torch.randn(d_model) * 0.05)

        # Self-Supervised Pre-Training Heads:
        # (a) 9-Slot Masked Clinical Symbol Classification Heads (5 classes: D, L, N, H, E)
        self.slot_cls_heads = nn.ModuleList([nn.Linear(d_model, 5) for _ in range(n_slots)])
        # (b) Continuous 9-Parameter Reconstruction Head
        self.cont_recon_head = nn.Linear(d_model, n_slots)
        # (c) Contrastive Projection Head
        self.contrast_proj = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.Linear(d_model, 128),
        )

    def forward_tokens(
        self,
        slot_ids: torch.Tensor,
        prior_vecs: torch.Tensor,
        cont_vecs: torch.Tensor,
        rhy_ids: torch.Tensor,
        run_ids: torch.Tensor,
        cmi_weights: torch.Tensor,
        pad_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode token sequence and return (token_states (B, L, D), pooled_repr (B, 2*D))."""
        B, L, _ = slot_ids.shape
        pos_idx = torch.arange(L, device=slot_ids.device).unsqueeze(0).expand(B, L)

        slot_sum = torch.zeros((B, L, self.d_model), dtype=prior_vecs.dtype, device=slot_ids.device)
        for s_i in range(self.n_slots):
            slot_sum = slot_sum + self.slot_embs[s_i](slot_ids[:, :, s_i])

        x = (
            (slot_sum / math.sqrt(self.n_slots))
            + self.prior_proj(prior_vecs)
            + 0.45 * self.cont_proj(cont_vecs)
            + 0.35 * self.rhythm_emb(rhy_ids)
            + 0.25 * self.run_emb(run_ids)
            + 0.25 * self.pos_emb(pos_idx)
        )
        x = self.in_norm(x)

        for layer in self.layers:
            x = layer(x, pad_mask=pad_mask, cmi_weights=cmi_weights)
        x = self.out_norm(x)

        # CMI-Weighted Self-Attention Pooling + Mean Pooling over valid tokens
        pool_logits = torch.matmul(x, self.attn_pool_query) / math.sqrt(self.d_model) + 2.2 * cmi_weights
        pool_logits = pool_logits.masked_fill(~pad_mask, -1e4)
        pool_weights = F.softmax(pool_logits, dim=-1).unsqueeze(-1)  # (B, L, 1)
        attn_pooled = torch.sum(x * pool_weights, dim=1)  # (B, D)

        valid_flt = pad_mask.unsqueeze(-1).float()
        mean_pooled = torch.sum(x * valid_flt, dim=1) / torch.clamp(torch.sum(valid_flt, dim=1), min=1.0)

        pooled = torch.cat([F.normalize(attn_pooled, dim=-1), F.normalize(mean_pooled, dim=-1)], dim=-1)
        return x, pooled


class RASPROTransformerFM:
    """High-level wrapper for training & encoding with RASPROTransformerBackbone on MPS/CUDA/CPU."""

    def __init__(
        self,
        n_slots: int = 9,
        d_prior: int = 72,
        d_model: int = 192,
        n_heads: int = 6,
        n_layers: int = 4,
        max_len: int = 8,
        prefer_device: str = "mps",
        seed: int = 42,
    ):
        self.n_slots = n_slots
        self.d_prior = d_prior
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.max_len = max_len
        self.seed = seed
        self.device = get_Compute_device(prefer_device)

        torch.manual_seed(seed)
        np.random.seed(seed)

        self.model = RASPROTransformerBackbone(
            n_slots=n_slots,
            d_prior=d_prior,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            max_len=max_len,
        ).to(self.device)
        self.cont_mean: Optional[np.ndarray] = None
        self.cont_std: Optional[np.ndarray] = None

    def _pack_tensors(
        self,
        slot_seqs: np.ndarray,
        prior_seqs: np.ndarray,
        cont_seqs: np.ndarray,
        rhy_seqs: np.ndarray,
        run_seqs: np.ndarray,
        cmi_seqs: np.ndarray,
        mask_seqs: np.ndarray,
    ) -> Tuple[torch.Tensor, ...]:
        """Normalize continuous features and move packed arrays to device."""
        if self.cont_mean is None or self.cont_std is None:
            valid = mask_seqs.astype(bool)
            valid_cont = cont_seqs[valid] if np.any(valid) else cont_seqs.reshape(-1, self.n_slots)
            self.cont_mean = np.mean(valid_cont, axis=0, keepdims=True).astype(np.float32)
            self.cont_std = (np.std(valid_cont, axis=0, keepdims=True) + 1e-4).astype(np.float32)

        cont_norm = np.clip((cont_seqs - self.cont_mean) / self.cont_std, -4.0, 4.0).astype(np.float32)
        return (
            torch.from_numpy(slot_seqs).long(),
            torch.from_numpy(prior_vecs := prior_seqs.astype(np.float32)),
            torch.from_numpy(cont_norm),
            torch.from_numpy(rhy_seqs).long(),
            torch.from_numpy(run_seqs).long(),
            torch.from_numpy(cmi_seqs.astype(np.float32)),
            torch.from_numpy(mask_seqs.astype(bool)),
        )

    def pretrain(
        self,
        slot_seqs: np.ndarray,
        prior_seqs: np.ndarray,
        cont_seqs: np.ndarray,
        rhy_seqs: np.ndarray,
        run_seqs: np.ndarray,
        cmi_seqs: np.ndarray,
        mask_seqs: np.ndarray,
        epochs: int = 18,
        batch_size: int = 256,
        lr: float = 1.5e-3,
    ) -> TorchPretrainStats:
        """Self-supervised pre-training via Masked Slot Modeling + Continuous Huber + InfoNCE."""
        self.cont_mean = None
        self.cont_std = None
        tensors = self._pack_tensors(slot_seqs, prior_seqs, cont_seqs, rhy_seqs, run_seqs, cmi_seqs, mask_seqs)
        t_slots, t_prior, t_cont, t_rhy, t_run, t_cmi, t_mask = [t.to(self.device) for t in tensors]

        n_records = t_slots.shape[0]
        avg_seq_len = float(t_mask.float().sum(dim=1).mean().item())

        optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=1e-2)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, epochs))

        self.model.train()
        if self.device.type == "mps":
            torch.mps.synchronize()
        t0 = time.perf_counter()

        initial_loss = 0.0
        final_loss = 0.0
        rng = torch.Generator(device="cpu")
        rng.manual_seed(self.seed)

        for ep in range(epochs):
            perm = torch.randperm(n_records, generator=rng).to(self.device)
            ep_loss_sum = 0.0
            n_batches = 0

            for start in range(0, n_records, batch_size):
                idx = perm[start : start + batch_size]
                b_slots = t_slots[idx]     # (B, L, S)
                b_prior = t_prior[idx]     # (B, L, d_prior)
                b_cont = t_cont[idx]       # (B, L, S)
                b_rhy = t_rhy[idx]         # (B, L)
                b_run = t_run[idx]         # (B, L)
                b_cmi = t_cmi[idx]         # (B, L)
                b_mask = t_mask[idx]       # (B, L)
                B, L, S = b_slots.shape

                # View 1: CMI-salience-weighted random slot & feature masking (30% mask rate)
                mask_prob = torch.clamp(0.25 + 0.30 * b_cmi.unsqueeze(-1), max=0.60)
                corrupt_mask = (torch.rand((B, L, S), device=self.device) < mask_prob) & b_mask.unsqueeze(-1)
                masked_slots_1 = torch.where(corrupt_mask, torch.full_like(b_slots, 5), b_slots)
                masked_cont_1 = torch.where(corrupt_mask, torch.zeros_like(b_cont), b_cont)
                win_drop_1 = (torch.rand((B, L, 1), device=self.device) < 0.25).float()
                masked_prior_1 = b_prior * (1.0 - 0.5 * win_drop_1)

                # View 2: Second stochastic view for InfoNCE contrastive alignment
                corrupt_mask_2 = (torch.rand((B, L, S), device=self.device) < 0.25) & b_mask.unsqueeze(-1)
                masked_slots_2 = torch.where(corrupt_mask_2, torch.full_like(b_slots, 5), b_slots)
                masked_cont_2 = torch.where(corrupt_mask_2, torch.zeros_like(b_cont), b_cont)

                tok_states_1, pooled_1 = self.model.forward_tokens(
                    masked_slots_1, masked_prior_1, masked_cont_1, b_rhy, b_run, b_cmi, b_mask
                )
                _, pooled_2 = self.model.forward_tokens(
                    masked_slots_2, b_prior, masked_cont_2, b_rhy, b_run, b_cmi, b_mask
                )

                # Loss 1: CMI-Salience-Weighted Masked Slot Cross-Entropy across all 9 slots
                salience_w = (1.0 + 2.5 * b_cmi) * b_mask.float()  # (B, L)
                loss_msm = torch.tensor(0.0, device=self.device)
                for s_i in range(S):
                    logits_s = self.model.slot_cls_heads[s_i](tok_states_1)  # (B, L, 5)
                    targets_s = torch.clamp(b_slots[:, :, s_i], 0, 4)        # (B, L)
                    ce_s = F.cross_entropy(logits_s.view(-1, 5), targets_s.view(-1), reduction="none").view(B, L)
                    active_s = corrupt_mask[:, :, s_i].float() + 0.15 * b_mask.float()
                    loss_msm = loss_msm + torch.sum(ce_s * salience_w * active_s) / torch.clamp(torch.sum(salience_w * active_s), min=1.0)
                loss_msm = loss_msm / S

                # Loss 2: Continuous 9-parameter physiological reconstruction (Smooth-L1)
                pred_cont = self.model.cont_recon_head(tok_states_1)  # (B, L, S)
                huber = F.smooth_l1_loss(pred_cont, b_cont, reduction="none")
                loss_cont = torch.sum(huber * corrupt_mask.float()) / torch.clamp(torch.sum(corrupt_mask.float()), min=1.0)

                # Loss 3: Contrastive InfoNCE loss between View 1 and View 2
                if B >= 4:
                    z1 = F.normalize(self.model.contrast_proj(pooled_1), dim=-1)
                    z2 = F.normalize(self.model.contrast_proj(pooled_2), dim=-1)
                    sim = torch.matmul(z1, z2.T) / 0.15
                    labels_cl = torch.arange(B, device=self.device)
                    loss_cl = 0.5 * (F.cross_entropy(sim, labels_cl) + F.cross_entropy(sim.T, labels_cl))
                else:
                    loss_cl = torch.tensor(0.0, device=self.device)

                loss = loss_msm + 0.50 * loss_cont + 0.25 * loss_cl
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.5)
                optimizer.step()

                ep_loss_sum += float(loss.item())
                n_batches += 1

            scheduler.step()
            ep_avg = ep_loss_sum / max(1, n_batches)
            if ep == 0:
                initial_loss = ep_avg
            final_loss = ep_avg

        if self.device.type == "mps":
            torch.mps.synchronize()
        elif self.device.type == "cuda":
            torch.cuda.synchronize()
        wall_time = time.perf_counter() - t0

        return TorchPretrainStats(
            device_name=str(self.device),
            n_records=n_records,
            n_epochs=epochs,
            avg_seq_len=avg_seq_len,
            wall_time_sec=wall_time,
            ms_per_epoch=(wall_time * 1000.0) / max(1, epochs),
            records_per_sec=(n_records * epochs) / max(1e-6, wall_time),
            initial_loss=initial_loss,
            final_loss=final_loss,
        )

    @torch.no_grad()
    def encode(
        self,
        slot_seqs: np.ndarray,
        prior_seqs: np.ndarray,
        cont_seqs: np.ndarray,
        rhy_seqs: np.ndarray,
        run_seqs: np.ndarray,
        cmi_seqs: np.ndarray,
        mask_seqs: np.ndarray,
        batch_size: int = 512,
    ) -> np.ndarray:
        """Encode dataset into frozen foundation representations (concatenating Transformer + Compositional Prior)."""
        self.model.eval()
        tensors = self._pack_tensors(slot_seqs, prior_seqs, cont_seqs, rhy_seqs, run_seqs, cmi_seqs, mask_seqs)
        t_slots, t_prior, t_cont, t_rhy, t_run, t_cmi, t_mask = tensors
        n_records = t_slots.shape[0]
        outputs = []

        for start in range(0, n_records, batch_size):
            end = min(n_records, start + batch_size)
            b_slots = t_slots[start:end].to(self.device)
            b_prior = t_prior[start:end].to(self.device)
            b_cont = t_cont[start:end].to(self.device)
            b_rhy = t_rhy[start:end].to(self.device)
            b_run = t_run[start:end].to(self.device)
            b_cmi = t_cmi[start:end].to(self.device)
            b_mask = t_mask[start:end].to(self.device)

            _, pooled = self.model.forward_tokens(b_slots, b_prior, b_cont, b_rhy, b_run, b_cmi, b_mask)

            # Also compute CMI-weighted pooled 72-D compositional prior & 9-D continuous summary
            w_logits = (2.5 * b_cmi).masked_fill(~b_mask, -1e4)
            w_attn = F.softmax(w_logits, dim=-1).unsqueeze(-1)
            prior_attn = F.normalize(torch.sum(b_prior * w_attn, dim=1), dim=-1)
            valid_f = b_mask.unsqueeze(-1).float()
            prior_mean = F.normalize(torch.sum(b_prior * valid_f, dim=1) / torch.clamp(torch.sum(valid_f, dim=1), min=1.0), dim=-1)
            cont_mean = F.normalize(torch.sum(b_cont * valid_f, dim=1) / torch.clamp(torch.sum(valid_f, dim=1), min=1.0), dim=-1)
            cont_max = F.normalize(torch.max(b_cont.masked_fill(~b_mask.unsqueeze(-1), -10.0), dim=1).values, dim=-1)

            combined = torch.cat([0.75 * pooled, 1.15 * prior_attn, 0.95 * prior_mean, 0.85 * cont_mean, 0.65 * cont_max], dim=-1)
            outputs.append(combined.cpu().numpy())

        return np.concatenate(outputs, axis=0).astype(np.float64)


class DeepPatchMAEBackbone(nn.Module):
    """ST-MEM / PatchTST style Multi-Layer Transformer Masked Autoencoder for L=25 12-Lead Patches."""

    def __init__(self, patch_dim: int = 36, d_model: int = 192, n_heads: int = 6, n_layers: int = 4, seq_len: int = 25):
        super().__init__()
        self.seq_len = seq_len
        self.d_model = d_model
        self.patch_proj = nn.Linear(patch_dim, d_model)
        self.mask_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.pos_emb = nn.Parameter(torch.randn(1, seq_len, d_model) * 0.02)
        self.in_norm = nn.LayerNorm(d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_model * 3, dropout=0.1, activation="gelu", batch_first=True, norm_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.out_norm = nn.LayerNorm(d_model)
        self.recon_head = nn.Linear(d_model, patch_dim)

    def forward(self, patches: torch.Tensor, mask_bool: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        B, L, _ = patches.shape
        x = self.patch_proj(patches)
        if mask_bool is not None:
            x = torch.where(mask_bool.unsqueeze(-1), self.mask_token.expand(B, L, self.d_model), x)
        x = self.in_norm(x + self.pos_emb[:, :L, :])
        x = self.out_norm(self.encoder(x))
        mean_pool = F.normalize(torch.mean(x, dim=1), dim=-1)
        max_pool = F.normalize(torch.max(x, dim=1).values, dim=-1)
        return x, torch.cat([mean_pool, max_pool], dim=-1)


class DeepPatchTransformerFM:
    """ST-MEM / PatchTST-style Multi-Layer PyTorch Masked Autoencoder Baseline on L=25 Patches."""

    def __init__(self, patch_dim: int = 36, d_model: int = 192, n_heads: int = 6, n_layers: int = 4, prefer_device: str = "mps", seed: int = 42):
        self.patch_dim = patch_dim
        self.d_model = d_model
        self.seed = seed
        self.device = get_Compute_device(prefer_device)
        torch.manual_seed(seed)
        self.model = DeepPatchMAEBackbone(patch_dim=patch_dim, d_model=d_model, n_heads=n_heads, n_layers=n_layers).to(self.device)

    def pretrain(self, patch_seqs: np.ndarray, epochs: int = 18, batch_size: int = 256, lr: float = 1.5e-3) -> TorchPretrainStats:
        t_patches = torch.from_numpy(patch_seqs.astype(np.float32)).to(self.device)
        n_records, seq_len, _ = t_patches.shape
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=1e-2)
        self.model.train()
        if self.device.type == "mps":
            torch.mps.synchronize()
        t0 = time.perf_counter()
        initial_loss = 0.0
        final_loss = 0.0
        rng = torch.Generator(device="cpu")
        rng.manual_seed(self.seed)

        for ep in range(epochs):
            perm = torch.randperm(n_records, generator=rng).to(self.device)
            ep_loss = 0.0
            n_b = 0
            for start in range(0, n_records, batch_size):
                idx = perm[start : start + batch_size]
                b_p = t_patches[idx]
                B, L, _ = b_p.shape
                m_bool = torch.rand((B, L), device=self.device) < 0.40
                states, _ = self.model(b_p, mask_bool=m_bool)
                pred = self.model.recon_head(states)
                mse = ((pred - b_p) ** 2).mean(dim=-1)
                loss = torch.sum(mse * m_bool.float()) / torch.clamp(torch.sum(m_bool.float()), min=1.0)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                ep_loss += float(loss.item())
                n_b += 1
            if ep == 0:
                initial_loss = ep_loss / max(1, n_b)
            final_loss = ep_loss / max(1, n_b)

        if self.device.type == "mps":
            torch.mps.synchronize()
        wall_time = time.perf_counter() - t0
        return TorchPretrainStats(
            device_name=str(self.device),
            n_records=n_records,
            n_epochs=epochs,
            avg_seq_len=float(seq_len),
            wall_time_sec=wall_time,
            ms_per_epoch=(wall_time * 1000.0) / max(1, epochs),
            records_per_sec=(n_records * epochs) / max(1e-6, wall_time),
            initial_loss=initial_loss,
            final_loss=final_loss,
        )

    @torch.no_grad()
    def encode(self, patch_seqs: np.ndarray, batch_size: int = 512) -> np.ndarray:
        self.model.eval()
        t_patches = torch.from_numpy(patch_seqs.astype(np.float32))
        n_records = t_patches.shape[0]
        outs = []
        for start in range(0, n_records, batch_size):
            b_p = t_patches[start : start + batch_size].to(self.device)
            _, pooled = self.model(b_p, mask_bool=None)
            outs.append(pooled.cpu().numpy())
        return np.concatenate(outs, axis=0).astype(np.float64)
