# WM/train_world_model.py
from __future__ import annotations

import argparse
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader




def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def list_npz_files(data_dir: Path) -> List[Path]:
    files = sorted(data_dir.glob("wm_ep*_gen*.npz"))
    return files

def build_block_causal_mask(Tctx: int, N: int, device: torch.device) -> torch.Tensor:
    """
    mask shape: (L, L) with True meaning "masked" (not allowed).
    Rule: token at time t can attend to any token at time t' <= t (full within same time).
    """
    L = Tctx * N
    t_idx = torch.arange(L, device=device) // N  # (L,)
    # allow if t_j <= t_i, so mask if t_j > t_i
    mask = (t_idx[None, :] > t_idx[:, None])  # (L, L) bool
    return mask

@dataclass
class Batch:
    # input
    x_tokens: torch.Tensor # (B, Tctx, N, D)
    x_actions: torch.Tensor # (B, Tctx, 9)
    time_valid: torch.Tensor # (B, Tctx) bool
    # targets
    y_tokens: torch.Tensor # (B, Tctx, N, D)
    y_rewards: Optional[torch.Tensor] # (B, Tctx)
    y_dones: Optional[torch.Tensor] # (B, Tctx)



# Dataset
class EpisodeChunkDataset(Dataset):
    """
    Each __getitem__ loads ONE episode file and returns ONE random chunk.
    Good enough for 9k episodes; keeps implementation simple and robust.
    """

    def __init__(
        self,
        files: List[Path],
        context_len: int,
        N: int = 19,
        D: int = 10,
        require_reward_done: bool = False,
        max_resample: int = 20,
    ):
        self.files = files
        self.context_len = int(context_len)
        self.N = int(N)
        self.D = int(D)
        self.require_reward_done = bool(require_reward_done)
        self.max_resample = int(max_resample)

    def __len__(self):
        return len(self.files)

    def _load_npz(self, p: Path):
        data = np.load(p, allow_pickle=False)
        
        # (T,19,10)
        tokens = data["tokens"].astype(np.float32)     
        # (T,9)     
        actions = data["marine_action"].astype(np.int64)    
        rewards = None
        dones = None
        if "rewards" in data.files:
            rewards = data["rewards"].astype(np.float32) # (T,)
        if "dones" in data.files:
            dones = data["dones"].astype(np.int8) # (T,)
        # print("dones:", dones)
        return tokens, actions, rewards, dones

    def __getitem__(self, idx: int):
        # resample if too short or missing reward/done when required
        for _ in range(self.max_resample):
            p = self.files[idx]
            tokens, actions, rewards, dones = self._load_npz(p)
            T = tokens.shape[0]

            if tokens.shape[1] != self.N or tokens.shape[2] != self.D:
                raise ValueError(f"!!!Unexpected token shape in {p}: {tokens.shape}")

            has_rd = (rewards is not None) and (dones is not None)
            if self.require_reward_done and not has_rd:
                idx = random.randrange(len(self.files))
                continue

            # need at least context_len+1 to form (x[t], y[t+1])
            need = self.context_len + 1
            if T < 2:
                idx = random.randrange(len(self.files))
                continue

            if T >= need:
                
                if random.random() < 0.5:
                    s = T - need  # 50% 
                else:
                    s = random.randrange(0, T - need + 1)
                
                chunk_tokens = tokens[s:s+need] # (need,N,D)
                chunk_actions = actions[s:s+need] # (need,9) actions for alignment, we will use first context_len
                if has_rd:
                    chunk_rewards = rewards[s:s+need] # (need,)
                    chunk_dones = dones[s:s+need] # (need,)
                else:
                    chunk_rewards = None
                    chunk_dones = None

                # x uses first context_len, y uses next context_len
                x_tokens = chunk_tokens[:-1] # (Tctx,N,D)
                y_tokens = chunk_tokens[1:]
                x_actions = chunk_actions[:-1] # (Tctx,9)

                time_valid = np.ones((self.context_len,), dtype=np.bool_) # all valid

                if has_rd:
                    y_rewards = chunk_rewards[:-1] # reward for transition at t
                    y_dones = chunk_dones[:-1]
                else:
                    y_rewards = None
                    y_dones = None

                return x_tokens, x_actions, time_valid, y_tokens, y_rewards, y_dones

            else:
                # pad if shorter than need
                # we pad time with zeros tokens; mark invalid steps
                # chunk length = T
                chunk_tokens = tokens
                chunk_actions = actions
                if has_rd:
                    chunk_rewards = rewards
                    chunk_dones = dones
                else:
                    chunk_rewards = None
                    chunk_dones = None

                # build padded arrays of length need
                pad_len = need - T
                pad_tokens = np.zeros((pad_len, self.N, self.D), dtype=np.float32)
                pad_actions = np.zeros((pad_len, 9), dtype=np.int64)
                chunk_tokens2 = np.concatenate([chunk_tokens, pad_tokens], axis=0)
                chunk_actions2 = np.concatenate([chunk_actions, pad_actions], axis=0)

                x_tokens = chunk_tokens2[:-1]
                y_tokens = chunk_tokens2[1:]
                x_actions = chunk_actions2[:-1]

                # valid steps only up to min(context_len, T-1)
                valid_steps = min(self.context_len, max(0, T-1))
                time_valid = np.zeros((self.context_len,), dtype=np.bool_)
                time_valid[:valid_steps] = True

                if has_rd:
                    # pad rewards/dones too
                    pad_r = np.zeros((pad_len,), dtype=np.float32)
                    pad_d = np.zeros((pad_len,), dtype=np.int8)
                    chunk_rewards2 = np.concatenate([chunk_rewards, pad_r], axis=0)
                    chunk_dones2 = np.concatenate([chunk_dones, pad_d], axis=0)
                    y_rewards = chunk_rewards2[:-1]
                    y_dones = chunk_dones2[:-1]
                else:
                    y_rewards = None
                    y_dones = None

                return x_tokens, x_actions, time_valid, y_tokens, y_rewards, y_dones

        raise RuntimeError("Failed to sample a valid episode chunk after resampling.")

def collate_fn(batch_list):
    # batch elements can have y_rewards/y_dones None
    x_tokens = torch.tensor(np.stack([b[0] for b in batch_list], axis=0), dtype=torch.float32)
    x_actions = torch.tensor(np.stack([b[1] for b in batch_list], axis=0), dtype=torch.long)
    time_valid = torch.tensor(np.stack([b[2] for b in batch_list], axis=0), dtype=torch.bool)
    y_tokens = torch.tensor(np.stack([b[3] for b in batch_list], axis=0), dtype=torch.float32)

    has_reward = batch_list[0][4] is not None
    has_done = batch_list[0][5] is not None
    y_rewards = None
    y_dones = None
    if has_reward and has_done:
        y_rewards = torch.tensor(np.stack([b[4] for b in batch_list], axis=0), dtype=torch.float32)
        y_dones = torch.tensor(np.stack([b[5] for b in batch_list], axis=0), dtype=torch.float32)

    return Batch(x_tokens=x_tokens, x_actions=x_actions, time_valid=time_valid,
                 y_tokens=y_tokens, y_rewards=y_rewards, y_dones=y_dones)



# Model
class SpacetimeTransformerWM(nn.Module):
    def __init__(
        self,
        N: int = 19,
        D: int = 10,
        num_actions: int = 9,
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 6,
        dropout: float = 0.1,
        context_len: int = 64,
        type_vocab_size: int = 8,
        side_vocab_size: int = 4,
        action_vocab_size: Optional[int] = None, # num_actions + 1(no_action)
        use_reward_done: bool = True,
    ):
        super().__init__()
        self.N = N
        self.D = D
        self.context_len = context_len
        self.num_actions = num_actions

        if action_vocab_size is None:
            action_vocab_size = num_actions + 1 # last = no_action
        self.action_noop_id = action_vocab_size - 1

        # token format: [type, side, alive, x,y,hp, dx,dy, distx,disty]
        self.type_emb = nn.Embedding(type_vocab_size, d_model)
        self.side_emb = nn.Embedding(side_vocab_size, d_model)
        self.action_emb = nn.Embedding(action_vocab_size, d_model)

        self.slot_emb = nn.Embedding(N, d_model)
        self.time_emb = nn.Embedding(context_len, d_model)

        # continuous: alive + 7 continuous = 8 dims
        self.cont_in = nn.Linear(1 + 7, d_model)

        self.in_ln = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=4*d_model,
            dropout=dropout, activation="gelu", batch_first=True, norm_first=True
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)

        # per-token prediction: alive_logit + 7 continuous
        self.token_head = nn.Linear(d_model, 1 + 7)

        # reward/done heads (per time step, after pooling over entities)
        self.use_reward_done = bool(use_reward_done)
        if self.use_reward_done:
            self.reward_head = nn.Sequential(
                nn.LayerNorm(d_model),
                nn.Linear(d_model, d_model),
                nn.GELU(),
                nn.Linear(d_model, 1),
            )
            self.done_head = nn.Sequential(
                nn.LayerNorm(d_model),
                nn.Linear(d_model, d_model),
                nn.GELU(),
                nn.Linear(d_model, 1),
            )

        # prebuild block-causal mask for max context_len
        self.register_buffer("_mask_full", build_block_causal_mask(context_len, N, device=torch.device("cpu")), persistent=False)

    @staticmethod
    def _safe_int(x: torch.Tensor) -> torch.Tensor:
        return torch.clamp(torch.round(x).long(), min=0)

    def forward(self, x_tokens: torch.Tensor, x_actions: torch.Tensor, time_valid: torch.Tensor):
        """
        x_tokens: (B,T,N,D)
        x_actions: (B,T,9)
        time_valid: (B,T) bool
        """
        # print("x_tokens:", x_tokens.shape)#[32, 64, 19, 10]
        # print("x_actions:", x_actions.shape)#[32, 64, 9]
        B, T, N, D = x_tokens.shape # B:32 T:64 N:19 D:10
        assert N == self.N and D == self.D

        device = x_tokens.device
        L = T * N # 64x19

        # parse discrete/continuous
        # discrete
        raw_type = x_tokens[..., 0] # float
        raw_side = x_tokens[..., 1]
        alive = x_tokens[..., 2] # float 0/1

        # Map raw_type/raw_side to indices:
        # Here we assume they are already small ints; unknowns will be clamped.
        type_id = self._safe_int(raw_type)
        side_id = self._safe_int(raw_side)

        # continuous (7 dims): x,y,hp,dx,dy,distx,disty
        cont = x_tokens[..., 3:10]

        # action ids per token (B,T,N)
        act_tok = torch.full((B, T, N), fill_value=self.action_noop_id, dtype=torch.long, device=device)
        # marines 0..8 get actions
        act_tok[:, :, :9] = x_actions[:, :, :9].clamp(min=0, max=self.num_actions - 1)

        # continuous pack: [alive, cont7]
        cont_in = torch.cat([alive.unsqueeze(-1), cont], dim=-1)  # (B,T,N,8)
        h_cont = self.cont_in(cont_in) # (B,T,N,d_model)

        # embeddings
        h = h_cont
        h = h + self.type_emb(type_id.clamp(max=self.type_emb.num_embeddings - 1))
        h = h + self.side_emb(side_id.clamp(max=self.side_emb.num_embeddings - 1))
        h = h + self.action_emb(act_tok.clamp(max=self.action_emb.num_embeddings - 1))

        # slot/time embeddings
        slot_ids = torch.arange(N, device=device).view(1, 1, N).expand(B, T, N)
        time_ids = torch.arange(T, device=device).view(1, T, 1).expand(B, T, N)
        # if T < context_len, time_emb ok; else you must init model with big enough context_len
        h = h + self.slot_emb(slot_ids)
        h = h + self.time_emb(time_ids)

        h = self.drop(self.in_ln(h))

        # flatten (B, L, d_model)
        h = h.reshape(B, L, -1)

        # key padding mask: pad entire time steps if invalid
        # time_valid: (B,T) -> (B,T,N) -> (B,L)
        pad_tok = (~time_valid).unsqueeze(-1).expand(B, T, N).reshape(B, L)

        # block-causal mask
        if self._mask_full.device != device:
            self._mask_full = self._mask_full.to(device)
        mask = self._mask_full[:L, :L]

        # transformer
        z = self.encoder(h, mask=mask, src_key_padding_mask=pad_tok) # (B,L,d_model)
        z = z.reshape(B, T, N, -1)

        # token prediction for next state (aligned outside)
        pred = self.token_head(z) # (B,T,N,8)
        pred_alive_logit = pred[..., 0]
        pred_cont = pred[..., 1:] # (B,T,N,7)

        out = {
            "pred_alive_logit": pred_alive_logit,
            "pred_cont": pred_cont,
        }

        if self.use_reward_done:
            # pool per time step
            pooled = z.mean(dim=2)  # (B,T,d_model)
            out["pred_reward"] = self.reward_head(pooled).squeeze(-1)  # (B,T)
            out["pred_done_logit"] = self.done_head(pooled).squeeze(-1)  # (B,T)

        return out



# Loss
def compute_losses(
    out: Dict[str, torch.Tensor],
    y_tokens: torch.Tensor, # (B,T,N,D)
    time_valid: torch.Tensor, # (B,T)
    y_rewards: Optional[torch.Tensor],
    y_dones: Optional[torch.Tensor],
    reward_done_enabled: bool,
    done_pos_weight: float = 5.0,
):
    """
    y_tokens format same as x_tokens: [type,side,alive,x,y,hp,dx,dy,distx,disty]
    We supervise:
      alive (BCE)
      continuous 7 dims (SmoothL1), masked by alive==1
      reward (SmoothL1)
      done (BCE), masked by time_valid
    """
    B, T, N, D = y_tokens.shape
    device = y_tokens.device

    # targets
    y_alive = y_tokens[..., 2] # (B,T,N)
    y_cont = y_tokens[..., 3:10] # (B,T,N,7)

    # masks
    tv = time_valid.float().unsqueeze(-1).unsqueeze(-1)# (B,T,1,1)
    alive_mask = (y_alive > 0.5).float().unsqueeze(-1) # (B,T,N,1)
    cont_mask = tv * alive_mask # (B,T,N,1)

    # alive loss (compute for all tokens in valid time)
    bce = nn.BCEWithLogitsLoss(reduction="none")
    alive_loss_raw = bce(out["pred_alive_logit"], y_alive) # (B,T,N)
    alive_loss = (alive_loss_raw * time_valid.float().unsqueeze(-1)).sum() / (time_valid.float().sum() * N + 1e-6)

    # cont loss (only when alive==1 and time_valid)
    smooth = nn.SmoothL1Loss(reduction="none")
    cont_loss_raw = smooth(out["pred_cont"], y_cont) # (B,T,N,7)
    cont_loss = (cont_loss_raw * cont_mask).sum() / (cont_mask.sum() * 7 + 1e-6)

    loss = alive_loss + cont_loss
    logs = {
        "loss_total": loss.detach().item(),
        "loss_alive": alive_loss.detach().item(),
        "loss_cont": cont_loss.detach().item(),
    }

    if reward_done_enabled and (y_rewards is not None) and (y_dones is not None):
        # reward
        r_loss_raw = smooth(out["pred_reward"], y_rewards) # (B,T)
        r_loss = (r_loss_raw * time_valid.float()).sum() / (time_valid.float().sum() + 1e-6)

        # done (pos_weight to fight sparsity)
        pos_w = torch.tensor([done_pos_weight], device=device, dtype=torch.float32)
        bce_done = nn.BCEWithLogitsLoss(reduction="none", pos_weight=pos_w)
        d_loss_raw = bce_done(out["pred_done_logit"], y_dones) # (B,T)
        d_loss = (d_loss_raw * time_valid.float()).sum() / (time_valid.float().sum() + 1e-6)

        loss = loss + 0.5 * r_loss + 0.5 * d_loss
        logs.update({
            "loss_total": loss.detach().item(),
            "loss_reward": r_loss.detach().item(),
            "loss_done": d_loss.detach().item(),
        })

    return loss, logs



# Train/Eval
@torch.no_grad()
def evaluate(model, loader, device, reward_done_enabled: bool):
    model.eval()
    agg = {}
    n = 0
    for batch in loader:
        batch = move_batch(batch, device)
        out = model(batch.x_tokens, batch.x_actions, batch.time_valid)
        loss, logs = compute_losses(out, batch.y_tokens, batch.time_valid, batch.y_rewards, batch.y_dones, reward_done_enabled)
        for k, v in logs.items():
            agg[k] = agg.get(k, 0.0) + float(v)
        n += 1
    for k in agg:
        agg[k] /= max(1, n)
    return agg

def move_batch(b: Batch, device: torch.device) -> Batch:
    return Batch(
        x_tokens=b.x_tokens.to(device),
        x_actions=b.x_actions.to(device),
        time_valid=b.time_valid.to(device),
        y_tokens=b.y_tokens.to(device),
        y_rewards=None if b.y_rewards is None else b.y_rewards.to(device),
        y_dones=None if b.y_dones is None else b.y_dones.to(device),
    )

def train_one_epoch(model, loader, optim, scaler, device, epoch, reward_done_enabled: bool, grad_clip: float = 1.0):
    model.train()
    agg = {}
    n = 0
    for step, batch in enumerate(loader):
        
        batch = move_batch(batch, device)
        optim.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast(enabled=(scaler is not None)):
            out = model(batch.x_tokens, batch.x_actions, batch.time_valid)
            
            loss, logs = compute_losses(out, batch.y_tokens, batch.time_valid, batch.y_rewards, batch.y_dones, reward_done_enabled)

        if scaler is not None:
            scaler.scale(loss).backward()
            if grad_clip > 0:
                scaler.unscale_(optim)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optim)
            scaler.update()
        else:
            loss.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optim.step()

        for k, v in logs.items():
            agg[k] = agg.get(k, 0.0) + float(v)
        n += 1

    for k in agg:
        agg[k] /= max(1, n)
    return agg



# Main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, default=r"C:\linux_project\SwarmTPG_clean\runs\DefeatZerglingsAndBanelings_WM_RSwarm_LocalMem\wm_data")
    ap.add_argument("--out_dir", type=str, default=r"C:\linux_project\SwarmTPG_clean\runs\DefeatZerglingsAndBanelings_WM_RSwarm_LocalMem\wm_ckpt")

    ap.add_argument("--context_len", type=int, default=64)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=0)

    ap.add_argument("--d_model", type=int, default=256)
    ap.add_argument("--nhead", type=int, default=8)
    ap.add_argument("--layers", type=int, default=6)
    ap.add_argument("--dropout", type=float, default=0.1)

    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--no_amp", action="store_true")

    ap.add_argument("--train_reward_done", default=True)
    ap.add_argument("--require_reward_done", default=True)

    args = ap.parse_args()

    set_seed(args.seed)

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    files = list_npz_files(data_dir)
    if not files:
        raise FileNotFoundError(f"No npz files found under {data_dir}")

    # quick check for rewards/dones presence
    sample = np.load(files[0], allow_pickle=False)
    has_rd = ("rewards" in sample.files) and ("dones" in sample.files)

    if args.train_reward_done and not has_rd:
        print("[WARN] enabled --train_reward_done but npz seems missing rewards/dones.")
        print(" recorder's save_episode() likely didn't save them. Reward/done head will be DISABLED.")
        train_rd = False
    else:
        train_rd = bool(args.train_reward_done and has_rd)

    # split
    rng = np.random.RandomState(args.seed)
    idxs = np.arange(len(files))
    rng.shuffle(idxs)
    
    split = int(0.95 * len(files))
    
    
    tr_files = [files[i] for i in idxs[:split]]
    va_files = [files[i] for i in idxs[split:]]

    train_ds = EpisodeChunkDataset(tr_files, context_len=args.context_len, require_reward_done=args.require_reward_done)
    val_ds = EpisodeChunkDataset(va_files, context_len=args.context_len, require_reward_done=args.require_reward_done)

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True,
        collate_fn=collate_fn, drop_last=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
        collate_fn=collate_fn, drop_last=False
    )

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")

    model = SpacetimeTransformerWM(
        N=19, D=10,
        num_actions=9,
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.layers,
        dropout=args.dropout,
        context_len=args.context_len,
        type_vocab_size=16,
        side_vocab_size=4,
        use_reward_done=train_rd,
    ).to(device)

    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = None if (args.no_amp or device.type != "cuda") else torch.cuda.amp.GradScaler()

    best_val = float("inf")
    best_path = out_dir / "wm_best.pt"
    last_path = out_dir / "wm_last.pt"

    print(f"[INFO] episodes: train={len(tr_files)} val={len(va_files)}")
    print(f"[INFO] train_reward_done: {train_rd} (npz_has_rd={has_rd})")
    print(f"[INFO] device={device} amp={'off' if scaler is None else 'on'}")

    for epoch in range(1, args.epochs + 1):
        tr_logs = train_one_epoch(model, train_loader, optim, scaler, device, epoch, reward_done_enabled=train_rd)
        va_logs = evaluate(model, val_loader, device, reward_done_enabled=train_rd)

        print(f"\n[Epoch {epoch:03d}] "
              f"train loss={tr_logs.get('loss_total', 0):.6f} "
              f"(alive={tr_logs.get('loss_alive', 0):.6f} cont={tr_logs.get('loss_cont', 0):.6f})  |  "
              f"val loss={va_logs.get('loss_total', 0):.6f} "
              f"(alive={va_logs.get('loss_alive', 0):.6f} cont={va_logs.get('loss_cont', 0):.6f})")

        if train_rd:
            print(f"train reward={tr_logs.get('loss_reward', 0):.6f} done={tr_logs.get('loss_done', 0):.6f}  |  "
                  f"val reward={va_logs.get('loss_reward', 0):.6f} done={va_logs.get('loss_done', 0):.6f}")

        # save last
        torch.save({
            "epoch": epoch,
            "model": model.state_dict(),
            "optim": optim.state_dict(),
            "args": vars(args),
        }, last_path)

        # save best
        v = float(va_logs.get("loss_total", 0.0))
        if v < best_val:
            best_val = v
            torch.save({
                "epoch": epoch,
                "model": model.state_dict(),
                "optim": optim.state_dict(),
                "args": vars(args),
                "best_val": best_val,
            }, best_path)
            print(f"[SAVE] best -> {best_path} (val={best_val:.6f})")

    print(f"\n[Done]!!! best_val={best_val:.6f}  best_ckpt={best_path}")


if __name__ == "__main__":
    main()
