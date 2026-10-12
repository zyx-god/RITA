# -*- coding: utf-8 -*-
"""
RITA 无掩码推理：对 <Generator>/<Real|Fake>/<Category>/xxx.jpg 结构的数据集
逐图输出"真假分数"，并按生成器汇总 AUC。

支持 GPU 批量推理 + FP16 + 多进程图片解码，用于 Kaggle 等环境。

用法:
  # 试点: 每个生成器 Real/Fake 各取 N 张
  python predict_realfake.py --ckpt best.pth --root <数据根目录> --per_gen 20 --out preds_pilot.csv --calibrate

  # 全量 (GPU)
  python predict_realfake.py --ckpt best.pth --root <数据根目录> --out preds_all.csv --batch 16 --workers 4

  # 单张图
  python predict_realfake.py --ckpt best.pth --single D:\\a.jpg
"""
import os, csv, time, random, argparse

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms as T

from rita import RITA

EXTS = ('.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tif', '.tiff')


def build_transform(size=512):
    return T.Compose([
        T.Resize((size, size), interpolation=T.InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])


@torch.no_grad()
def ar_infer_batch(model, images, max_steps=3, eos_ratio=0.9, amp=True):
    """images: (B,3,512,512) 已归一化。

    AR 的每一步在 batch 内并行；已经输出 EOS 的样本被冻结，不再更新其 mask。
    返回 (step0_stats, steps_taken)，step0_stats 是各统计量的 (B,) 张量。

    第 0 步输出的语义是"哪些像素被篡改"(类别 0)，压成标量即为 image-level 假图分数。
    """
    model.eval()
    B = images.shape[0]
    dev = images.device
    H, W = images.shape[-2:]
    eos_id = model.eos_token_id
    pad_id = model.num_classes - 1

    current_mask = torch.full((B, H, W), pad_id, dtype=torch.long, device=dev)
    iter_val = 0.0
    active = torch.ones(B, dtype=torch.bool, device=dev)
    steps_taken = torch.zeros(B, dtype=torch.long, device=dev)
    step0 = None

    use_amp = amp and dev.type == 'cuda'

    for k in range(max_steps):
        iter_channel = torch.full((B, H, W), iter_val, dtype=torch.float32, device=dev)
        inp = torch.cat([images,
                         current_mask.float().unsqueeze(1),
                         iter_channel.unsqueeze(1)], dim=1)      # (B,5,H,W)

        with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=use_amp):
            logits = model(inp)
        logits = logits.float()

        prob = F.softmax(logits, dim=1)
        pred = logits.argmax(dim=1)                              # (B,H,W)

        p_tamper = prob[:, 0]
        eos_frac = (pred == eos_id).float().mean(dim=(1, 2))

        if k == 0:
            step0 = {
                'p_tamper_mean': p_tamper.mean(dim=(1, 2)).cpu(),
                'tamper_ratio': (pred == 0).float().mean(dim=(1, 2)).cpu(),
                'p_tamper_p95': torch.quantile(p_tamper.flatten(1), 0.95, dim=1).cpu(),
            }

        steps_taken[active] = k + 1
        active = active & (eos_frac < eos_ratio)
        if not bool(active.any()):
            break
        current_mask[active] = pred[active]
        iter_val += 1.0

    return step0, steps_taken


class ImageListDataset(torch.utils.data.Dataset):
    """按索引返回预处理后的张量；解码失败返回全零张量并带回错误信息。"""

    def __init__(self, items, transform):
        self.items = items
        self.tf = transform

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        _, _, _, path = self.items[i]
        try:
            x = self.tf(Image.open(path).convert('RGB'))
            return i, x, ''
        except Exception as e:
            return i, torch.zeros(3, 512, 512), str(e)


def scan(root, per_gen=None, seed=0):
    """扫描 <Generator>/<Real|Fake>/<Category>/img。per_gen 为每个生成器的采样上限。"""
    rng = random.Random(seed)
    buckets = {}
    for gen in sorted(os.listdir(root)):
        gdir = os.path.join(root, gen)
        if not os.path.isdir(gdir):
            continue
        for split in ('Real', 'Fake'):
            sdir = os.path.join(gdir, split)
            if not os.path.isdir(sdir):
                continue
            paths = []
            for cat in sorted(os.listdir(sdir)):
                cdir = os.path.join(sdir, cat)
                if not os.path.isdir(cdir):
                    continue
                for fn in sorted(os.listdir(cdir)):
                    if fn.lower().endswith(EXTS):
                        paths.append((gen, split, cat, os.path.join(cdir, fn)))
            if per_gen is not None:
                half = max(1, per_gen // 2)
                if len(paths) > half:
                    paths = rng.sample(paths, half)
            buckets[(gen, split)] = paths
    items = []
    for k in sorted(buckets):
        items.extend(buckets[k])
    return items


def auc(scores, labels):
    """Mann-Whitney U 版 AUC，不依赖 sklearn"""
    data = sorted(zip(scores, labels))
    n_pos = sum(labels)
    n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float('nan')
    ranks = [0.0] * len(data)
    i = 0
    while i < len(data):
        j = i
        while j + 1 < len(data) and data[j + 1][0] == data[i][0]:
            j += 1
        avg = (i + j) / 2.0 + 1
        for k in range(i, j + 1):
            ranks[k] = avg
        i = j + 1
    s = sum(ranks[k] for k in range(len(data)) if data[k][1] == 1)
    return (s - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def best_threshold(scores, labels):
    best = (0.5, -1.0)
    for t in sorted(set(scores)):
        acc = sum((1 if s > t else 0) == y for s, y in zip(scores, labels)) / len(labels)
        if acc > best[1]:
            best = (t, acc)
    return best


def load_model(ckpt, device):
    model = RITA(num_classes=4)
    ck = torch.load(ckpt, map_location='cpu')
    if isinstance(ck, dict) and 'state_dict' in ck:
        ck = ck['state_dict']
    ck = {k.replace('module.', ''): v for k, v in ck.items()}
    missing, unexpected = model.load_state_dict(ck, strict=False)
    print(f'[ckpt] missing={len(missing)} unexpected={len(unexpected)}')
    if len(missing) > 20 or len(unexpected) > 20:
        print('  !! 权重与模型结构不匹配，结果很可能无效')
        print('  missing 样例:', missing[:5])
    return model.to(device).eval()


def report(rows, args):
    labels = [r['is_fake'] for r in rows]
    print('\n=== 每个生成器 AUC (分数=p_tamper_mean) ===')
    aucs = []
    for g in sorted({r['generator'] for r in rows}):
        sub = [r for r in rows if r['generator'] == g]
        y = [r['is_fake'] for r in sub]
        fk = [r['p_tamper_mean'] for r in sub if r['is_fake'] == 1]
        rl = [r['p_tamper_mean'] for r in sub if r['is_fake'] == 0]
        a = auc([r['p_tamper_mean'] for r in sub], y)
        if a == a:
            aucs.append(a)
        print(f'  {g:22s} n={len(sub):5d}  AUC={a:.4f}  '
              f'假图均分={sum(fk)/max(1,len(fk)):.3f}  真图均分={sum(rl)/max(1,len(rl)):.3f}')
    if aucs:
        print(f'  平均 AUC = {sum(aucs)/len(aucs):.4f}  ({len(aucs)} 个生成器可算)')

    print('\n=== 整体 ===')
    s_all = [r['p_tamper_mean'] for r in rows]
    print(f"  AUC(p_tamper_mean) = {auc(s_all, labels):.4f}")
    print(f"  AUC(tamper_ratio)  = {auc([r['tamper_ratio'] for r in rows], labels):.4f}")
    th = args.threshold
    if args.calibrate:
        th, acc = best_threshold(s_all, labels)
        print(f'  [校准] 最优阈值={th:.4f}  ACC={acc:.4f}')
    if th is not None:
        acc = sum((1 if s > th else 0) == y for s, y in zip(s_all, labels)) / len(labels)
        print(f'  阈值 {th} 下的 ACC = {acc:.4f}')
    print('\n注意: Real 图在每个生成器下重复出现，整体 ACC 会被放大，请以各生成器 AUC 为准。')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--root')
    ap.add_argument('--single')
    ap.add_argument('--out', default='preds.csv')
    ap.add_argument('--per_gen', type=int, default=None, help='每个生成器采样上限(Real/Fake 各一半)')
    ap.add_argument('--max_steps', type=int, default=3)
    ap.add_argument('--eos_ratio', type=float, default=0.9)
    ap.add_argument('--batch', type=int, default=8)
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--threads', type=int, default=0, help='CPU 线程数，0=不设置')
    ap.add_argument('--no_amp', action='store_true', help='禁用 FP16')
    ap.add_argument('--threshold', type=float, default=None)
    ap.add_argument('--calibrate', action='store_true')
    args = ap.parse_args()

    if args.threads:
        torch.set_num_threads(args.threads)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'[device] {device}  batch={args.batch}  workers={args.workers}')
    if device.type == 'cuda':
        print(f'[gpu] {torch.cuda.get_device_name(0)}')

    model = load_model(args.ckpt, device)
    tf = build_transform(512)

    if args.single:
        x = tf(Image.open(args.single).convert('RGB')).unsqueeze(0).to(device)
        s0, _ = ar_infer_batch(model, x, args.max_steps, args.eos_ratio, not args.no_amp)
        th = args.threshold if args.threshold is not None else 0.5
        print(f"{args.single}\n  判定: {'FAKE' if s0['tamper_ratio'][0] > th else 'REAL'}"
              f"  p_tamper_mean={s0['p_tamper_mean'][0]:.4f}"
              f"  tamper_ratio={s0['tamper_ratio'][0]:.4f}")
        return

    items = scan(args.root, args.per_gen)
    print(f'[data] {len(items)} 张')

    ds = ImageListDataset(items, tf)
    loader = torch.utils.data.DataLoader(
        ds, batch_size=args.batch, shuffle=False,
        num_workers=args.workers, pin_memory=(device.type == 'cuda'),
        persistent_workers=(args.workers > 0),
    )

    rows, done, t_start = [], 0, time.time()
    for idxs, xs, errs in loader:
        xs = xs.to(device, non_blocking=True)
        s0, steps = ar_infer_batch(model, xs, args.max_steps, args.eos_ratio, not args.no_amp)
        for j, i in enumerate(idxs.tolist()):
            done += 1
            if errs[j]:
                print(f'  [skip] {items[i][3]}: {errs[j]}')
                continue
            gen, split, cat, path = items[i]
            rows.append({
                'generator': gen, 'split': split, 'is_fake': 1 if split == 'Fake' else 0,
                'category': cat, 'path': path,
                'p_tamper_mean': s0['p_tamper_mean'][j].item(),
                'tamper_ratio': s0['tamper_ratio'][j].item(),
                'p_tamper_p95': s0['p_tamper_p95'][j].item(),
                'n_steps': int(steps[j]),
            })
        if done % (args.batch * 4) < args.batch or done == len(items):
            el = time.time() - t_start
            print(f'  {done}/{len(items)}  {el/done:.2f}s/张  ETA {el/done*(len(items)-done)/60:.1f}min',
                  flush=True)

    if not rows:
        print('没有成功的样本'); return
    with open(args.out, 'w', newline='', encoding='utf-8-sig') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f'[saved] {args.out} ({len(rows)} 行)  总耗时 {(time.time()-t_start)/60:.1f} 分钟')
    report(rows, args)


if __name__ == '__main__':
    main()