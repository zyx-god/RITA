# -*- coding: utf-8 -*-
"""
从完整数据集里按生成器抽取小子集，保持 <Generator>/<Real|Fake>/<Category>/ 结构不变，
这样 predict_realfake.py 不用改就能直接用。

抽样是分层的：每个类别轮流取，避免只抽到某一类。

用法:
  python make_subset.py --root "C:\\Users\\86186\\Desktop\\dataset_classified" \
                        --out "D:\\rita_subset" --per_split 20
"""
import os, argparse, random, shutil, zipfile

EXTS = ('.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tif', '.tiff')


def stratified_sample(paths_by_cat, n, rng):
    """每个类别轮流取，直到凑够 n 张。"""
    cats = sorted(paths_by_cat)
    for c in cats:
        rng.shuffle(paths_by_cat[c])
    picked, i = [], 0
    while len(picked) < n and any(paths_by_cat[c] for c in cats):
        c = cats[i % len(cats)]
        if paths_by_cat[c]:
            picked.append(paths_by_cat[c].pop())
        i += 1
    return picked


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', required=True, help='完整数据集根目录')
    ap.add_argument('--out', required=True, help='输出子集目录')
    ap.add_argument('--per_split', type=int, default=20,
                    help='每个生成器每个 split(Real/Fake) 抽多少张')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--zip', action='store_true',
                    help='同时打包成 zip (用 Python zipfile，条目名保证是正斜杠)')
    args = ap.parse_args()

    rng = random.Random(args.seed)
    total, total_bytes = 0, 0
    summary = []

    for gen in sorted(os.listdir(args.root)):
        gdir = os.path.join(args.root, gen)
        if not os.path.isdir(gdir):
            continue
        for split in ('Real', 'Fake'):
            sdir = os.path.join(gdir, split)
            if not os.path.isdir(sdir):
                continue

            by_cat = {}
            for cat in sorted(os.listdir(sdir)):
                cdir = os.path.join(sdir, cat)
                if not os.path.isdir(cdir):
                    continue
                fs = [os.path.join(cdir, f) for f in sorted(os.listdir(cdir))
                      if f.lower().endswith(EXTS)]
                if fs:
                    by_cat[cat] = fs
            if not by_cat:
                continue

            picked = stratified_sample(by_cat, args.per_split, rng)
            for src in picked:
                cat = os.path.basename(os.path.dirname(src))
                dst_dir = os.path.join(args.out, gen, split, cat)
                os.makedirs(dst_dir, exist_ok=True)
                dst = os.path.join(dst_dir, os.path.basename(src))
                shutil.copy2(src, dst)
                total += 1
                total_bytes += os.path.getsize(src)
            summary.append((gen, split, len(picked)))

    print(f'\n{"生成器":<24}{"split":<8}{"抽取":>6}')
    for g, s, n in summary:
        print(f'{g:<24}{s:<8}{n:>6}')
    print('-' * 40)
    print(f'合计 {total} 张, {total_bytes/1024**2:.1f} MB')
    print(f'输出目录: {args.out}')

    if args.zip:
        zip_path = args.out.rstrip('\\/') + '.zip'
        if os.path.exists(zip_path):
            os.remove(zip_path)
        # 注意: 必须用 zipfile 而不是 PowerShell 的 Compress-Archive，
        # 后者的实现在 Windows 上会把反斜杠写进条目名，Kaggle 会拒收。
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED, compresslevel=1) as z:
            for dp, _, fns in os.walk(args.out):
                for f in sorted(fns):
                    full = os.path.join(dp, f)
                    rel = os.path.relpath(full, args.out).replace(os.sep, '/')
                    z.write(full, rel)
        with zipfile.ZipFile(zip_path) as z:
            names = z.namelist()
            bad = [x for x in names if '\\' in x]
        size_mb = os.path.getsize(zip_path) / 1024 ** 2
        print(f'\n已打包: {zip_path}')
        print(f'  条目数 {len(names)}, 大小 {size_mb:.1f} MB, 含反斜杠条目 {len(bad)}')
        print(f'  前 3 条: {names[:3]}')
        assert not bad, 'zip 条目名里出现反斜杠，上传会被 Kaggle 拒绝'
        print('  OK — 可以直接上传')
    else:
        print('\n下次可以直接加 --zip 打包，例如:')
        print(f'  python make_subset.py --root <数据集> --out "{args.out}" --per_split 20 --zip')


if __name__ == '__main__':
    main()