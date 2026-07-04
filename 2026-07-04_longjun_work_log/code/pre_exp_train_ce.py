"""
预实验 · 第四步：CE baseline 训练
  setting in {clean, noise20, noise40}
  模型选择用 val_acc；test 仅在训练结束后用 best-val 模型评估一次（防 test 调参/泄漏）。
产物目录 experiments/pre_exp/ce_<setting>/：config.yaml train.log metrics.json best_model.pth run_command.txt
"""
import os, sys, json, time, argparse
import numpy as np, torch, torch.nn as nn
import yaml
sys.path.insert(0, os.path.expanduser('~/aa_workspace/code'))
from pre_exp_common import EXP, N_CLS, SimpleCNN1D, make_loaders, evaluate


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--setting', required=True, choices=['clean', 'noise20', 'noise40'])
    ap.add_argument('--epochs', type=int, default=30)
    ap.add_argument('--batch', type=int, default=128)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()

    out_dir = f'{EXP}/ce_{args.setting}'
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
    lossf = nn.CrossEntropyLoss()

    cfg = {'setting': args.setting, 'model': 'SimpleCNN1D', 'epochs': args.epochs,
           'batch': args.batch, 'lr': args.lr, 'seed': args.seed, 'optimizer': 'Adam',
           'loss': 'CrossEntropy', 'data_source': 'B: preprocess_features 4096-dim',
           'n_classes': N_CLS, 'eval_policy': 'select by val_acc; test evaluated once at end'}
    yaml.safe_dump(cfg, open(f'{out_dir}/config.yaml', 'w'), allow_unicode=True)
    open(f'{out_dir}/run_command.txt', 'w').write(
        f'cd ~/aa_workspace && source .venv/bin/activate && CUDA_VISIBLE_DEVICES=0 '
        f'python code/pre_exp_train_ce.py --setting {args.setting} --epochs {args.epochs} '
        f'--batch {args.batch} --lr {args.lr} --seed {args.seed}\n')

    log(f"setting={args.setting} dev={dev} "
        f"({torch.cuda.get_device_name(0) if dev == 'cuda' else ''})")
    log(f"params={sum(p.numel() for p in model.parameters()) / 1e6:.3f}M train_batches={len(tr_dl)}")

    best_va, best_ep = -1.0, -1
    t0 = time.time()
    for ep in range(args.epochs):
        model.train(); tl = 0
        for xb, yb in tr_dl:
            xb, yb = xb.to(dev), yb.to(dev)
            opt.zero_grad(); loss = lossf(model(xb), yb); loss.backward(); opt.step()
            tl += loss.item() * len(yb)
        va = evaluate(model, va_dl, dev)
        star = '★' if va['accuracy'] > best_va else ' '
        if va['accuracy'] > best_va:
            best_va = va['accuracy']; best_ep = ep
            torch.save({'model': model.state_dict(), 'ep': ep, 'val_acc': va['accuracy']},
                       f'{out_dir}/best_model.pth')
        log(f"{star} ep{ep:02d} tr_loss={tl / len(tr_dl.dataset):.4f} "
            f"val_acc={va['accuracy']:.3f} val_bacc={va['balanced_accuracy']:.3f} (best_va={best_va:.3f})")

    # 训练后：加载 best-val 模型，test 评估一次
    model.load_state_dict(torch.load(f'{out_dir}/best_model.pth')['model'])
    te = evaluate(model, te_dl, dev)
    metrics = {'setting': args.setting, 'method': 'CE',
               'best_epoch': best_ep, 'best_val_acc': best_va,
               'test': te, 'time_s': round(time.time() - t0, 1)}
    json.dump(metrics, open(f'{out_dir}/metrics.json', 'w'), indent=2, ensure_ascii=False)
    log(f"DONE best_ep={best_ep} best_val_acc={best_va:.3f} | TEST(@best-val)={te}")
    logf.close()


if __name__ == '__main__':
    main()
