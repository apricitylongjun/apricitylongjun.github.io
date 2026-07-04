"""
第二轮预实验入口：强 backbone(ResNet1D) 下的 CE / robust。
输出 experiments/pre_exp2/，不覆盖第一轮(pre_exp/, SimpleCNN1D)。
模型选择仍用 val_acc；test 仅末尾用 best-val 模型评估一次。
用法见各 run_command.txt。
"""
import os, sys, json, time, argparse
import numpy as np, torch, torch.nn as nn
import torch.nn.functional as F
import yaml
sys.path.insert(0, os.path.expanduser('~/aa_workspace/code'))
from pre_exp_common import N_CLS, SimpleCNN1D, ResNet1D, make_loaders, evaluate


def build_model(name):
    return SimpleCNN1D() if name == 'cnn' else ResNet1D()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default='resnet1d', choices=['cnn', 'resnet1d'])
    ap.add_argument('--method', default='ce', choices=['ce', 'robust'])
    ap.add_argument('--setting', required=True, choices=['clean', 'noise20', 'noise40'])
    ap.add_argument('--epochs', type=int, default=40)
    ap.add_argument('--batch', type=int, default=128)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--warmup', type=int, default=10)
    ap.add_argument('--lam_cons', type=float, default=0.5)
    ap.add_argument('--noise_std', type=float, default=0.1)
    ap.add_argument('--w_floor', type=float, default=0.1)
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()

    OUT = os.path.expanduser('~/aa_workspace/experiments/pre_exp2')
    out_dir = f'{OUT}/{args.model}_{args.method}_{args.setting}'
    os.makedirs(out_dir, exist_ok=True)
    logf = open(f'{out_dir}/train.log', 'w')

    def log(m):
        line = f"[{time.strftime('%H:%M:%S')}] {m}"
        print(line, flush=True); logf.write(line + '\n'); logf.flush()

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    tr_dl, va_dl, te_dl = make_loaders(args.batch, args.setting)
    model = build_model(args.model).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    cfg = vars(args).copy()
    cfg.update({'n_classes': N_CLS, 'eval_policy': 'select by val_acc; test once at end',
                'data_source': 'B: preprocess_features 4096-dim (同第一轮)'})
    yaml.safe_dump(cfg, open(f'{out_dir}/config.yaml', 'w'), allow_unicode=True)
    open(f'{out_dir}/run_command.txt', 'w').write(
        f'cd ~/aa_workspace && source .venv/bin/activate && CUDA_VISIBLE_DEVICES=0 '
        f'python code/pre_exp2_train.py ' + ' '.join(f'--{k} {v}' for k, v in vars(args).items()) + '\n')

    log(f"model={args.model} method={args.method} setting={args.setting} dev={dev} "
        f"params={sum(p.numel() for p in model.parameters()) / 1e6:.3f}M")
    best_va, best_ep = -1.0, -1
    t0 = time.time()
    for ep in range(args.epochs):
        model.train(); tl = 0
        robust_on = (args.method == 'robust') and (ep >= args.warmup)
        for xb, yb in tr_dl:
            xb, yb = xb.to(dev), yb.to(dev)
            opt.zero_grad()
            out = model(xb)
            loss_per = F.cross_entropy(out, yb, reduction='none')
            if robust_on:
                with torch.no_grad():
                    l = loss_per.detach()
                    w = (1.0 - (l - l.min()) / (l.max() - l.min() + 1e-6)).clamp(min=args.w_floor)
                ce = (w * loss_per).mean()
                aug = xb + torch.randn_like(xb) * args.noise_std
                p = F.softmax(out, dim=1).detach()
                q = F.log_softmax(model(aug), dim=1)
                cons = F.kl_div(q, p, reduction='batchmean')
                loss = ce + args.lam_cons * cons
            else:
                loss = loss_per.mean()
            loss.backward(); opt.step(); tl += loss.item() * len(yb)
        va = evaluate(model, va_dl, dev)
        star = '★' if va['accuracy'] > best_va else ' '
        if va['accuracy'] > best_va:
            best_va = va['accuracy']; best_ep = ep
            torch.save({'model': model.state_dict(), 'ep': ep}, f'{out_dir}/best_model.pth')
        tag = 'robust' if robust_on else ('warmup' if args.method == 'robust' else 'ce')
        log(f"{star} ep{ep:02d}[{tag:6s}] loss={tl/len(tr_dl.dataset):.4f} "
            f"val_acc={va['accuracy']:.3f} val_bacc={va['balanced_accuracy']:.3f} (best={best_va:.3f})")

    model.load_state_dict(torch.load(f'{out_dir}/best_model.pth')['model'])
    te = evaluate(model, te_dl, dev)
    metrics = {'model': args.model, 'method': args.method, 'setting': args.setting,
               'best_epoch': best_ep, 'best_val_acc': best_va, 'test': te,
               'time_s': round(time.time() - t0, 1)}
    json.dump(metrics, open(f'{out_dir}/metrics.json', 'w'), indent=2, ensure_ascii=False)
    log(f"DONE best_ep={best_ep} val={best_va:.3f} | TEST(@best-val)={te}")
    logf.close()


if __name__ == '__main__':
    main()
