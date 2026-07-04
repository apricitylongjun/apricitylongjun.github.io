"""
预实验 · 第五步：简单鲁棒方法
  setting in {noise20, noise40}（clean 无噪声，不需要鲁棒）

方法（满足 spec 5 点）：
  1. warm-up 前 W epoch 用标准 CE
  2. warmup 后，按每个样本的 CE loss 估计可靠性（loss 小 = 更可能干净）
  3. 对低可靠（高 loss）样本降低 CE 权重：w = 1 - 归一化loss，下限 w_floor
  4. 一致性约束：x 加小高斯噪声增强 -> aug；要求 p(aug) 与 teacher p(x) 一致（KL）
  5. 不引入复杂模块，单网络、单 forward+一次增强 forward

模型选择仍用 val_acc；test 仅在末尾用 best-val 模型评估一次。
产物 experiments/pre_exp/ours_<setting>/：config.yaml train.log metrics.json best_model.pth run_command.txt
"""
import os, sys, json, time, argparse
import numpy as np, torch, torch.nn as nn
import torch.nn.functional as F
import yaml
sys.path.insert(0, os.path.expanduser('~/aa_workspace/code'))
from pre_exp_common import EXP, N_CLS, SimpleCNN1D, make_loaders, evaluate


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--setting', required=True, choices=['noise20', 'noise40'])
    ap.add_argument('--epochs', type=int, default=30)
    ap.add_argument('--batch', type=int, default=128)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--warmup', type=int, default=10, help='前 warmup epoch 纯 CE')
    ap.add_argument('--lam_cons', type=float, default=0.5, help='一致性正则系数')
    ap.add_argument('--noise_std', type=float, default=0.1, help='一致性增强高斯噪声 std')
    ap.add_argument('--w_floor', type=float, default=0.1, help='样本权重下限(防噪声样本权重塌缩到0)')
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()

    out_dir = f'{EXP}/ours_{args.setting}'
    os.makedirs(out_dir, exist_ok=True)
    logf = open(f'{out_dir}/train.log', 'w')

    def log(m):
        line = f"[{time.strftime('%H:%M:%S')}] {m}"
        print(line, flush=True); logf.write(line + '\n'); logf.flush()

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    tr_dl, va_dl, te_dl = make_loaders(args.batch, args.setting)
    model = SimpleCNN1D().to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    cfg = vars(args).copy()
    cfg.update({'model': 'SimpleCNN1D',
                'method': 'robust = warmup-CE + loss-reliability-weight + consistency-KL',
                'n_classes': N_CLS, 'data_source': 'B: preprocess_features 4096-dim',
                'eval_policy': 'select by val_acc; test evaluated once at end'})
    yaml.safe_dump(cfg, open(f'{out_dir}/config.yaml', 'w'), allow_unicode=True)
    open(f'{out_dir}/run_command.txt', 'w').write(
        f'cd ~/aa_workspace && source .venv/bin/activate && CUDA_VISIBLE_DEVICES=0 '
        f'python code/pre_exp_train_robust.py ' + ' '.join(f'--{k} {v}' for k, v in vars(args).items()) + '\n')

    log(f"setting={args.setting} dev={dev} warmup={args.warmup} lam_cons={args.lam_cons} noise_std={args.noise_std} w_floor={args.w_floor}")
    best_va, best_ep = -1.0, -1
    t0 = time.time()
    for ep in range(args.epochs):
        model.train(); tl = 0; wsum = 0
        robust_on = ep >= args.warmup
        for xb, yb in tr_dl:
            xb, yb = xb.to(dev), yb.to(dev)
            opt.zero_grad()
            out = model(xb)
            loss_per = F.cross_entropy(out, yb, reduction='none')   # 每样本 CE
            if robust_on:
                with torch.no_grad():
                    l = loss_per.detach()
                    # 可靠性权重：loss 小 -> 权重大（batch 内 min-max 归一化）
                    w = 1.0 - (l - l.min()) / (l.max() - l.min() + 1e-6)
                    w = w.clamp(min=args.w_floor)
                ce = (w * loss_per).mean()
                # 一致性：增强(加噪)后预测与原始(teacher)一致
                aug = xb + torch.randn_like(xb) * args.noise_std
                p = F.softmax(out, dim=1).detach()                  # teacher：原始预测
                q = F.log_softmax(model(aug), dim=1)
                cons = F.kl_div(q, p, reduction='batchmean')
                loss = ce + args.lam_cons * cons
                wsum += w.mean().item() * len(yb)
            else:
                loss = loss_per.mean()
                wsum += 1.0 * len(yb)
            loss.backward(); opt.step(); tl += loss.item() * len(yb)
        va = evaluate(model, va_dl, dev)
        star = '★' if va['accuracy'] > best_va else ' '
        if va['accuracy'] > best_va:
            best_va = va['accuracy']; best_ep = ep
            torch.save({'model': model.state_dict(), 'ep': ep, 'val_acc': va['accuracy']},
                       f'{out_dir}/best_model.pth')
        tag = 'robust' if robust_on else 'warmup'
        log(f"{star} ep{ep:02d}[{tag:6s}] loss={tl/len(tr_dl.dataset):.4f} "
            f"val_acc={va['accuracy']:.3f} val_bacc={va['balanced_accuracy']:.3f} "
            f"mean_w={wsum/len(tr_dl.dataset):.3f} (best={best_va:.3f})")

    model.load_state_dict(torch.load(f'{out_dir}/best_model.pth')['model'])
    te = evaluate(model, te_dl, dev)
    metrics = {'setting': args.setting, 'method': 'ours_robust',
               'best_epoch': best_ep, 'best_val_acc': best_va,
               'test': te, 'time_s': round(time.time() - t0, 1)}
    json.dump(metrics, open(f'{out_dir}/metrics.json', 'w'), indent=2, ensure_ascii=False)
    log(f"DONE best_ep={best_ep} best_val_acc={best_va:.3f} | TEST(@best-val)={te}")
    logf.close()


if __name__ == '__main__':
    main()
