# WM/test_world_model.py
from __future__ import annotations
from pathlib import Path
import numpy as np
import torch


from train_world_model import (
    SpacetimeTransformerWM, EpisodeChunkDataset, collate_fn, Batch, move_batch,
)

def sigmoid(x):
    return 1 / (1 + torch.exp(-x))

@torch.no_grad()
def one_step_eval(model, loader, device, has_rd: bool, alive_th: float = 0.5, done_th: float = 0.5):
    model.eval()

    alive_correct = 0.0
    alive_total = 0.0

    cont_abs_sum = 0.0
    cont_cnt = 0.0

    r_abs_sum = 0.0
    r_cnt = 0.0

    d_tp = 0.0
    d_fp = 0.0
    d_fn = 0.0

    for batch in loader:
        
        batch = move_batch(batch, device)
        out = model(batch.x_tokens, batch.x_actions, batch.time_valid)

        y = batch.y_tokens # (B,T,N,D)
        tv = batch.time_valid # (B,T) bool
        B, T, N, D = y.shape

        y_alive = y[..., 2] # (B,T,N)
        # print("y_alive:", y_alive)#[32, 64, 19]
        y_cont = y[..., 3:10] # (B,T,N,7)

        # alive acc (valid time only, across ALL entities)
        # pred/gt as bool
        pred_alive = (sigmoid(out["pred_alive_logit"]) > alive_th)# (B,T,N) bool
        gt_alive = (y_alive > 0.5) # (B,T,N) bool

        mask_time = tv.float()  # (B,T)
        # broadcast to (B,T,N)
        mask_tok = mask_time.unsqueeze(-1) # (B,T,1)

        alive_correct += ((pred_alive == gt_alive).float() * mask_tok).sum().item()
        alive_total += (mask_time.sum().item() * N)

        # cont MAE only when alive==1 and time_valid
        # mask: (B,T,N,1)
        cont_mask = mask_time.unsqueeze(-1).unsqueeze(-1) * gt_alive.float().unsqueeze(-1)
        abs_err = (out["pred_cont"] - y_cont).abs()# (B,T,N,7)

        cont_abs_sum += (abs_err * cont_mask).sum().item()
        cont_cnt += (cont_mask.sum().item() * 7.0)

        # reward/done eval only when has_rd and time_valid and GT reward/done exist
        if has_rd and (batch.y_rewards is not None) and (batch.y_dones is not None):
            # reward MAE (time_valid only)
            r_abs_sum += ((out["pred_reward"] - batch.y_rewards).abs() * mask_time).sum().item()
            r_cnt += mask_time.sum().item()

            # done F1 at threshold done_th (time_valid only)
            pred_done = (sigmoid(out["pred_done_logit"]) > done_th)# (B,T) bool
            gt_done = (batch.y_dones > 0.5)# (B,T) bool
            m = tv  # (B,T) bool

            d_tp += (pred_done & gt_done & m).sum().item()
            d_fp += (pred_done & (~gt_done) & m).sum().item()
            d_fn += ((~pred_done) & gt_done & m).sum().item()

    alive_acc = alive_correct / max(1e-9, alive_total)
    cont_mae = cont_abs_sum / max(1e-9, cont_cnt)

    # done positive rate
    pos = ((batch.y_dones > 0.5).float() * mask_time).sum().item()
    tot = mask_time.sum().item()
    print("pos/tot:", pos/tot)
    
    outm = {
        "alive_acc": float(alive_acc),
        "cont_mae_alive_only": float(cont_mae),
    }

    if has_rd and r_cnt > 0:
        outm["reward_mae"] = float(r_abs_sum / max(1e-9, r_cnt))

        prec = d_tp / max(1e-9, (d_tp + d_fp))
        rec= d_tp / max(1e-9, (d_tp + d_fn))
        f1= 2 * prec * rec / max(1e-9, (prec + rec))

        outm[f"done_precision@{done_th}"] = float(prec)
        outm[f"done_recall@{done_th}"] = float(rec)
        outm[f"done_f1@{done_th}"] = float(f1)

    return outm

@torch.no_grad()
def rollout_open_loop(model, ep_npz: Path, device, context_len=64, steps=200):
    """
    Open-loop: Use the token predicted by the model itself as the input for the next step of the rollout.

    """
    data = np.load(ep_npz, allow_pickle=False)
    tokens = data["tokens"].astype(np.float32) # (T,19,10)
    actions = data["marine_action"].astype(np.int64)# (T,9)

    T = tokens.shape[0]
    N, D = tokens.shape[1], tokens.shape[2]
    assert N == 19 and D == 10

    #context
    t0 = min(context_len, T - 1)
    ctx_tokens = torch.tensor(tokens[:t0], device=device).unsqueeze(0) # (1,t0,N,D)
    ctx_actions = torch.tensor(actions[:t0], device=device).unsqueeze(0) # (1,t0,9)
    time_valid = torch.ones((1, t0), device=device, dtype=torch.bool)

    # compare to GT
    mae_list = []
    alive_acc_list = []

    cur_tokens = ctx_tokens.clone()

    for k in range(steps):
        t_ctx = cur_tokens.shape[1]
        # lengthen context if needed (but not exceed context_len)
        if t_ctx > context_len:
            cur_tokens = cur_tokens[:, -context_len:]
            ctx_actions = ctx_actions[:, -context_len:]
            time_valid = time_valid[:, -context_len:]
            t_ctx = context_len

        out = model(cur_tokens, ctx_actions, time_valid)
        # next predict
        pred_alive_logit = out["pred_alive_logit"][:, -1] # (1,N)
        pred_cont = out["pred_cont"][:, -1] # (1,N,7)

        pred_alive = (sigmoid(pred_alive_logit) > 0.5).float()  # (1,N)

        # next token：type/side
        last = cur_tokens[:, -1]  # (1,N,D)
        next_tok = torch.zeros((1, N, D), device=device, dtype=torch.float32)
        next_tok[..., 0:2] = last[..., 0:2] # type, side
        next_tok[..., 2] = pred_alive # alive
        next_tok[..., 3:10] = pred_cont# cont 7 dims

        # compare to GT
        gt_t = t0 + k + 1
        if gt_t < T:
            gt = torch.tensor(tokens[gt_t], device=device).unsqueeze(0)  # (1,N,D)
            gt_alive = (gt[..., 2] > 0.5).float()

            alive_acc = (pred_alive == gt_alive).float().mean().item()
            # cont MAE only when GT alive==1
            mask = gt_alive.unsqueeze(-1)
            mae = ((pred_cont - gt[..., 3:10]).abs() * mask).sum().item() / (mask.sum().item() * 7 + 1e-6)

            alive_acc_list.append(alive_acc)
            mae_list.append(mae)

        # append predicted token
        cur_tokens = torch.cat([cur_tokens, next_tok.unsqueeze(1)], dim=1)

        # append action for next step（用 GT action）
        if (t0 + k + 1) < actions.shape[0]:
            a = torch.tensor(actions[t0 + k + 1], device=device).view(1,1,9)
        else:
            a = torch.zeros((1,1,9), device=device, dtype=torch.long)
        ctx_actions = torch.cat([ctx_actions, a], dim=1)
        time_valid = torch.cat([time_valid, torch.ones((1,1), device=device, dtype=torch.bool)], dim=1)

    return {
        "ep": str(ep_npz.name),
        "openloop_alive_acc_mean": float(np.mean(alive_acc_list)) if alive_acc_list else None,
        "openloop_cont_mae_mean": float(np.mean(mae_list)) if mae_list else None,
        "horizon": len(mae_list),
    }

def main():
    DATA_DIR = Path(r"C:\linux_project\SwarmTPG_clean\runs\DefeatZerglingsAndBanelings_WM_RSwarm_LocalMem\wm_data")
    CKPT_PATH = Path(r"C:\linux_project\SwarmTPG_clean\runs\DefeatZerglingsAndBanelings_WM_RSwarm_LocalMem\wm_ckpt\wm_best.pt")
    context_len = 64

    files = sorted(DATA_DIR.glob("wm_ep*_gen*.npz"))
    assert files, "no npz files"

    ckpt = torch.load(CKPT_PATH, map_location="cpu")
    args = ckpt.get("args", {})
    use_rd = bool(args.get("train_reward_done", False))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SpacetimeTransformerWM(
        N=19, D=10, num_actions=9,
        d_model=args.get("d_model", 256),
        nhead=args.get("nhead", 8),
        num_layers=args.get("layers", 6),
        dropout=args.get("dropout", 0.1),
        context_len=args.get("context_len", context_len),
        type_vocab_size=16, side_vocab_size=4,
        use_reward_done=use_rd,
    ).to(device)
    model.load_state_dict(ckpt["model"], strict=True)

    # one-step eval on random chunks
    rng = np.random.RandomState(0)
    rng.shuffle(files)
    split = int(0.95 * len(files))
    val_files = files[split:]

    ds = EpisodeChunkDataset(val_files, context_len=context_len, require_reward_done=use_rd)
    loader = torch.utils.data.DataLoader(ds, batch_size=1, shuffle=False, num_workers=0, collate_fn=collate_fn)

    metrics = one_step_eval(model, loader, device, has_rd=use_rd)
    print("[One-step metrics]")
    for k,v in metrics.items():
        print(f"  {k}: {v}")

    # open-loop rollout on a few episodes
    print("\n[Open-loop rollout]")
    for p in val_files[:5]:
        m = rollout_open_loop(model, p, device, context_len=context_len, steps=200)
        print(m)

if __name__ == "__main__":
    main()
