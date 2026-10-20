# -*- coding: utf-8 -*-
"""
抽取小子集，并统一规范化为 <generator>/<Real|Fake>/<Category>/，
这样 predict_realfake.py 的 scan() 不用改就能直接用。

支持两种源布局:
  A) GenImage 风格:  <root>/<Generator>/<Real|Fake>/<Category>/img
  B) SAIDO 风格:     <root>/<real|fake>/<Generator>/<Category>/img

也支持直接读取官方划分文件 (TSV)，只从某个划分抽样。

用法:
  # 用官方 test 划分（推荐, SAIDO 数据集）
  python make_subset.py --root "D:\\...\\dataset_v2" ^
      --split_file "D:\\...\\dataset_v2\\captions\\test.txt" ^
      --out "C:\\Users\\86186\\Desktop\\rita_subset" --per_split 100 --zip

  # 不用划分文件, 按目录抽样 (GenImage 风格)
  python make_subset.py --root "C:\\...\\dataset_classified" ^
      --out "C:\\...\\rita_subset" --per_split 20 --zip
"""
import os, re, argparse, random, shutil, zipfile

EXTS = ('.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tif', '.tiff')


def norm_split(s):
    return 'Fake' if s.lower() in ('fake', 'f') else 'Real'


def detect_layout(root):
    top = [d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))]
    if any(d.lower() in ('real', 'fake') for d in top):
        return 'split_first'
    return 'gen_first'


def collect_from_dirs(root):
    """返回 [(gen, split, relpath, fullpath)]，relpath 相对该 (gen, split) 根目录。"""
    items = []
    layout = detect_layout(root)
    print(f'[layout] {layout}')

    if layout == 'split_first':
        for split_dir in ('real', 'fake'):
            sdir = os.path.join(root, split_dir)
            if not os.path.isdir(sdir):
                continue
            for gen in sorted(os.listdir(sdir)):
                gdir = os.path.join(sdir, gen)
                if not os.path.isdir(gdir):
                    continue
                for dp, _, fns in os.walk(gdir):
                    for f in fns:
                        if f.lower().endswith(EXTS):
                            full = os.path.join(dp, f)
                            rel = os.path.relpath(full, gdir)
                            items.append((gen, norm_split(split_dir), rel, full))
    else:
        for gen in sorted(os.listdir(root)):
            gdir = os.path.join(root, gen)
            if not os.path.isdir(gdir):
                continue
            for split_dir in os.listdir(gdir):
                if split_dir.lower() not in ('real', 'fake'):
                    continue
                sdir = os.path.join(gdir, split_dir)
                if not os.path.isdir(sdir):
                    continue
                for dp, _, fns in os.walk(sdir):
                    for f in fns:
                        if f.lower().endswith(EXTS):
                            full = os.path.join(dp, f)
                            rel = os.path.relpath(full, sdir)
                            items.append((gen, norm_split(split_dir), rel, full))
    return items


def collect_from_tsv(root, tsv):
    """从官方划分文件收集。TSV 需含 path 列。"""
    lines = open(tsv, encoding='utf-8').read().splitlines()
    hdr = lines[0].split('\t')
    i_path = hdr.index('path')
    items = []
    for line in lines[1:]:
        cols = line.split('\t')
        if len(cols) <= i_path:
            continue
        raw = cols[i_path].replace('\\', '/')
        parts = [x for x in raw.split('/') if x not in ('', '.')]
        idx, split = None, None
        for key, sp in (('fake', 'Fake'), ('real', 'Real')):
            if key in parts:
                idx, split = parts.index(key), sp
                break
        if idx is None or idx + 1 >= len(parts):
            continue
        gen = parts[idx + 1]
        rest = parts[idx + 2:]
        full = os.path.join(root, *parts[idx:])
        if not os.path.isfile(full):
            continue
        items.append((gen, split, os.path.join(*rest) if rest else os.path.basename(full), full))
    return items


def stratified_sample(records, n, rng):
    """按 relpath 的第一层目录(类别)分层，轮流取，直到凑够 n。"""
    by_cat = {}
    for rec in records:
        cat = rec[1].replace('\\', '/').split('/')[0] if '/' in rec[1].replace('\\', '/') else ''
        by_cat.setdefault(cat, []).append(rec)
    cats = sorted(by_cat)
    for c in cats:
        rng.shuffle(by_cat[c])
    picked, i = [], 0
    while len(picked) < n and any(by_cat[c] for c in cats):
        c = cats[i % len(cats)]
        if by_cat[c]:
            picked.append(by_cat[c].pop())
        i += 1
    return picked


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--per_split', type=int, default=100,
                    help='每个生成器每个 split 的上限; 不足则全取')
    ap.add_argument('--split_file', default=None, help='官方划分 TSV (如 captions/test.txt)')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--zip', action='store_true')
    args = ap.parse_args()

    rng = random.Random(args.seed)
    items = (collect_from_tsv(args.root, args.split_file) if args.split_file
             else collect_from_dirs(args.root))
    print(f'[source] 共 {len(items)} 张')
    if args.split_file:
        print(f'[split_file] {args.split_file}')

    groups = {}
    for gen, split, rel, full in items:
        groups.setdefault((gen, split), []).append((rel, full))

    total, total_bytes, summary = 0, 0, []
    for (gen, split) in sorted(groups):
        picked = stratified_sample(groups[(gen, split)], args.per_split, rng)
        for rel, full in picked:
            dst = os.path.join(args.out, gen, split, *rel.replace('\\', '/').split('/'))
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.copy2(full, dst)
            total += 1
            total_bytes += os.path.getsize(full)
        summary.append((gen, split, len(picked), len(groups[(gen, split)])))

    print(f'\n{"生成器":<16}{"split":<7}{"抽取":>6}{"可用":>7}')
    for g, s, n, av in summary:
        print(f'{g:<16}{s:<7}{n:>6}{av:>7}')
    print('-' * 38)
    print(f'合计 {total} 张, {total_bytes/1024**2:.1f} MB')
    print(f'输出目录: {args.out}')

    if args.zip:
        zip_path = args.out.rstrip('\\/') + '.zip'
        if os.path.exists(zip_path):
            os.remove(zip_path)
        # 必须用 zipfile；PowerShell 的 Compress-Archive 在 Windows 上会写入反斜杠，
        # Kaggle 会以 "contains a forbidden character in name ('\\')" 拒收。
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED, compresslevel=1) as z:
            for dp, _, fns in os.walk(args.out):
                for f in sorted(fns):
                    full = os.path.join(dp, f)
                    rel = os.path.relpath(full, args.out).replace(os.sep, '/')
                    z.write(full, rel)
        with zipfile.ZipFile(zip_path) as z:
            names = z.namelist()
            bad = [x for x in names if '\\' in x]
        print(f'\n已打包: {zip_path}')
        print(f'  条目数 {len(names)}, 大小 {os.path.getsize(zip_path)/1024**2:.1f} MB, '
              f'含反斜杠条目 {len(bad)}')
        print(f'  前 3 条: {names[:3]}')
        assert not bad, 'zip 条目名里出现反斜杠，上传会被 Kaggle 拒绝'
        print('  OK — 可以直接上传')
    else:
        print('\n下次可以加 --zip 直接打包')


if __name__ == '__main__':
    main()