"""
第二阶段训练入口：ResNet1D + {CE, SCE, GCE, Co-teaching}
输出 experiments/pre_exp_stage2/{method}_{setting}_seed{N}/（5 文件）
- test 保持干净；模型选择只用 val；test 仅 best-val 末尾评一次
- Co-teaching：双网络交叉选小-loss 样本；报告 f1 网络（best_model.pth 存 f1）
"""
import os, sys, json, time, argparse
import numpy as np, torch, torch.nn as nn
import torch.nn.functional as F
import yaml
sys.path.insert(0, os.path.expanduser('~/aa_workspace/code'))
from pre_exp_common import N_CLS, ResNet1D, make_loaders, evaluate, sce_loss, gce_loss

OUT = os.path.expanduser('~/aa_workspace/experiments/pre_exp_stage2')
NOISE_RATE = {'clean': 0.0, 'noise20': 0.2, 'noise40': 0.4}


def loss_single(out, yb, method):
    if method == 'ce':
        return F.cross_entropy(out, yb)
    if method == 'sce':
        return sce_loss(out, yb)
    if method == 'gce':
        return gce_loss(out, yb)
    raise ValueError(method)


def coteaching_step(f1, f2, xb, yb, opt1, opt2, epoch, warmup, noise_est):
    """两网络各自按 loss 选小-loss 样本交叉喂对方；forget rate 随 epoch 线性升到 noise_est。"""
    out1, out2 = f1(xb), f2(xb)
    l1 = F.cross_entropy(out1, yb, reduction='none')
    l2 = F.cross_entropy(out2, yb, reduction='none')
    forget = min(epoch / max(warmup, 1) * noise_est, noise_est)
    remember_rate = max(1.0 - forget, 1e-2)
    k = max(int(remember_rate * len(yb)), 1)
    idx1 = l1.argsort()[:k]   # f1 认为干净的 -> 喂给 f2
    idx2 = l2.argsort()[:k]   # f2 认为干净的 -> 喂给 f1
    loss = F.cross_entropy(out1[idx2], yb[idx2]) + F.cross_entropy(out2[idx1], yb[idx1])
    opt1.zero_grad(); opt2.zero_grad(); loss.backward(); opt1.step(); opt2.step()
    return loss.item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--method', required=True, choices=['ce', 'sce', 'gce', 'coteaching'])
    ap.add_argument('--setting', required=True, choices=['clean', 'noise20', 'noise40'])
    ap.add_argument('--seed', type=int, required=True)
    ap.add_argument('--epochs', type=int, default=40)
    ap.add_argument('--batch', type=int, default=128)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--ct_warmup', type=int, default=10, help='co-teaching forget schedule warmup epoch')
    args = ap.parse_args()
    noise_est = NOISE_RATE[args.setting]

    sub = f'{args.method}_{args.setting}_seed{args.seed}'
    out_dir = f'{OUT}/{sub}'
    os.makedirs(out_dir, exist_ok=True)
    logf = open(f'{out_dir}/train.log', 'w')

    def log(m):
        line = f"[{time.strftime('%H:%M:%S')}] {m}"
        print(line, flush=True); logf.write(line + '\n'); logf.flush()

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    tr_dl, va_dl, te_dl = make_loaders(args.batch, args.setting)
    f1 = ResNet1D().to(dev)
    f2 = ResNet1D().to(dev) if args.method == 'coteaching' else None
    opt1 = torch.optim.Adam(f1.parameters(), lr=args.lr)
    opt2 = torch.optim.Adam(f2.parameters(), lr=args.lr) if f2 is not None else None

    cfg = vars(args).copy()
    cfg.update({'model': 'ResNet1D', 'n_classes': N_CLS, 'noise_rate': noise_est,
                'eval_policy': 'select by val_acc; test once at end',
                'coteaching_reports': 'f1 network'})
    yaml.safe_dump(cfg, open(f'{out_dir}/config.yaml', 'w'), allow_unicode=True)
    open(f'{out_dir}/run_command.txt', 'w').write(
        f'cd ~/aa_workspace && source .venv/bin/activate && CUDA_VISIBLE_DEVICES=0 '
        f'python code/pre_exp_stage2_train.py ' + ' '.join(f'--{k} {v}' for k, v in vars(args).items()) + '\n')
    log(f"method={args.method} setting={args.setting} seed={args.seed} dev={dev} "
        f"params={sum(p.numel() for p in f1.parameters())/1e6:.3f}M noise_est={noise_est}")

    best_va, best_ep = -1.0, -1
    t0 = time.time()
    for ep in range(args.epochs):
        f1.train()
        if f2 is not None:
            f2.train()
        tl = 0
        for xb, yb in tr_dl:
            xb, yb = xb.to(dev), yb.to(dev)
            if args.method == 'coteaching':
                tl += coteaching_step(f1, f2, xb, yb, opt1, opt2, ep, args.ct_warmup, noise_est) * len(yb)
            else:
                opt1.zero_grad()
                loss = loss_single(f1(xb), yb, args.method)
                loss.backward(); opt1.step()
                tl += loss.item() * len(yb)
        va = evaluate(f1, va_dl, dev)
        star = '★' if va['accuracy'] > best_va else ' '
        if va['accuracy'] > best_va:
            best_va = va['accuracy']; best_ep = ep
            torch.save({'model': f1.state_dict(), 'ep': ep}, f'{out_dir}/best_model.pth')
        log(f"{star} ep{ep:02d} loss={tl/len(tr_dl.dataset):.4f} val_acc={va['accuracy']:.3f} (best={best_va:.3f})")

    f1.load_state_dict(torch.load(f'{out_dir}/best_model.pth')['model'])
    te = evaluate(f1, te_dl, dev)
    metrics = {'backbone': 'ResNet1D', 'method': args.method, 'noise_rate': noise_est,
               'setting': args.setting, 'seed': args.seed,
               'best_epoch': best_ep, 'best_val_acc': best_va, 'test': te,
               'time_s': round(time.time() - t0, 1)}
    json.dump(metrics, open(f'{out_dir}/metrics.json', 'w'), indent=2, ensure_ascii=False)
    log(f"DONE best_ep={best_ep} val={best_va:.3f} | TEST(@best-val)={te}")
    logf.close()


if __name__ == '__main__':
    main()
